import os
import string
from collections import defaultdict

import msgpack
import numpy as np
import ray
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt_qa.flows.worldtree_qa.utils import table_store_mapping
from bayes_opt_qa.tasks.answer_selection_cvxpy import (AnswerSelectionCVXPY,
                                                       GraphConstructionCVXPY)
from bayes_opt_qa.tasks.eval_tasks import ExplanationEvaluationTask
from bayes_opt_qa.tasks.explanation_construction import EntityExtractionTask
from bayes_opt_qa.tasks.search.bm25 import BM25FitTask, BM25SearchTask
from bayes_opt_qa.tasks.utils import WorldTreeLemmatizer
from dynaconf import settings
from loguru import logger
from poly_nlp.tasks.datasets.genericskb import GenericsKBExtractionTask
from poly_nlp.tasks.datasets.worldtree.evaluation_tasks import \
    WorldTreeMAPEvaluationTask
from poly_nlp.tasks.datasets.worldtree.extraction_tasks import (
    TableStoreExtractionTask, WorldTreeExtractionTask, WorldTreeVersion)
from prefect import Flow, tags, task
from prefect.engine.results import LocalResult
from sklearn.gaussian_process.kernels import (RBF, RationalQuadratic,
                                              WhiteKernel)

ray.init()

logger.add("answer_selection_cvxpy_worldtree_dev.log")

WORLDTREE_VERSION = WorldTreeVersion.WorldTree_V1
version = "v1"


# Setup the setting
train_path = settings[f"worldtree_{version}"]["train"]
dev_path = settings[f"worldtree_{version}"]["dev"]
test_path = settings[f"worldtree_{version}"]["test"]
table_store_path = settings[f"worldtree_{version}"]["table_store_path"]
lemmatizer_path = settings[f"worldtree_{version}"]["lemmatizer_path"]

prediction_dev = settings[f"worldtree_{version}"]["unification"]["prediction_dev"]
prediction_train = settings[f"worldtree_{version}"]["unification"]["prediction_train"]
prediction_test = settings[f"worldtree_{version}"]["unification"]["prediction_test"]

generics_kb_path = settings["generics_kb"]["best"]

explanationlp_output = settings[f"worldtree_{version}"]["explanationlp"]

model_dir = settings["model_dir"]
checkpoint_dir = settings["checkpoint_dir"]

# Setup result handlers

TASK_NAME = f"worldtree_{WORLDTREE_VERSION.value}_final"


@task
def read_msgpack(file_name, limit, worldtree):
    with open(file_name, "rb") as f:
        expl = msgpack.unpackb(f.read(), raw=False)
    updated = {}
    for id, exps in expl.items():
        q_id, _ = id.split("|")
        if q_id in worldtree:
            updated[id] = {
                x: y for index, (x, y) in enumerate(exps.items()) if index < limit
            }

    return updated


@task
def write_msgpack(file_name, data):
    with open(file_name, "wb") as f:
        f.write(msgpack.packb(data, use_bin_type=False))


def eval_explanations(
    dataset, indexes: np.array, preds: np.array, mode, eval_params, **kwargs
):
    ids = list(map(lambda index: dataset.get_id(index), indexes))
    score_mapping = defaultdict(lambda: {})
    for index, (q_id, t_id) in enumerate(ids):
        score_mapping[q_id][t_id] = preds[index]
    selected_explanations = {}
    for q_id in score_mapping:
        sorted_facts = {
            k: v
            for k, v in sorted(score_mapping[q_id].items(), key=lambda item: item[1])
        }
        selected_explanations[q_id] = list(reversed(list(sorted_facts.keys())))
    gold_explanations = {
        id: list(question_exp["explanation"].keys())
        for id, question_exp in eval_params[mode].items()
    }
    map_score = eval_task.run(selected_explanations, gold_explanations)
    return map_score


@task
def worldtree_answer_mapping(worldtree, no_answer=False):
    return {
        f"{id}|{choice}": f"{question_exp['question']} {choice}"
        if not no_answer
        else f"{question_exp['question']}"
        for id, question_exp in worldtree.items()
        for choice in question_exp["choices"].values()
    }


@task
def worldtree_fold_mapping(worldtree, no_answer=False):
    new_worldtree = {
        id: question_exp
        for id, question_exp in worldtree.items()
        # if question_exp["examName"] == "NYSEDREGENTS" and question_exp["grade"] == 4
    }
    return new_worldtree


@task
def kind_of_mapping(table_store):
    return {
        id: fact["explanation"]["HYPONYM"]
        for id, fact in table_store.items()
        if fact["table_name"] == "KINDOF"
    }


@task
def expl_eval_mapping(expl, worldtree, limit):
    updated = {}
    for id, exps in expl.items():
        q_id, choice = id.split("|")
        if q_id in worldtree and worldtree[q_id]["answer"] == choice:
            updated[q_id] = list(exps.keys())[:limit]

    return updated


@task
def filter_facts(table_store, results):
    filtered_tablestore = {}
    for q_id, res in results.items():
        for t_id in res:
            filtered_tablestore[t_id] = table_store[t_id]
    return filtered_tablestore


LIMIT = 30

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)

# Initate the tasks
worldtree_extraction = WorldTreeExtractionTask(**cache_args)
table_store_extraction_task = TableStoreExtractionTask(**cache_args)
generics_kb_extraction_task = GenericsKBExtractionTask(**cache_args)
build_search_index = BM25FitTask(**cache_args)
bm25_search_task = BM25SearchTask(**cache_args)
lemmatizer_task = WorldTreeLemmatizer(**cache_args)
entity_extraction = EntityExtractionTask(**cache_args)
graph_extraction = GraphConstructionCVXPY(**cache_args)
answer_selection = AnswerSelectionCVXPY()


explanation_evaluate = ExplanationEvaluationTask()
eval_task = WorldTreeMAPEvaluationTask()

eval_task_baseline = WorldTreeMAPEvaluationTask()


prediction = prediction_dev

# NOTE run export PREFECT__FLOWS__CHECKPOINTING=true


def explanation_construction(**opts):
    # Build the flow
    with Flow("Question answering with WorldTreeCorpus") as flow:
        with tags("dev"):
            worldtree_dev = worldtree_extraction(
                dev_path,
                WORLDTREE_VERSION,
                skip_missing_explanations=True,
                skip_empty_explanations=False,
            )
            worldtree = worldtree_fold_mapping(worldtree_dev)
            lemmatized_question = lemmatizer_task(
                worldtree_answer_mapping(worldtree, no_answer=False), lemmatizer_path
            )
            dev_entities = entity_extraction(lemmatized_question)
        # worldtree_train = worldtree_extraction(
        #     train_path,
        #     WORLDTREE_VERSION,
        #     skip_missing_explanations=True,
        #     skip_empty_explanations=False,
        # )
        # worldtree_test = worldtree_extraction(
        #     test_path, WORLDTREE_VERSION, skip_empty_explanations=True
        # )
        with tags("table_store"):
            table_store = table_store_extraction_task(
                table_store_path, original_map=True
            )
            lemmatized_facts = lemmatizer_task(
                table_store_mapping(table_store), lemmatizer_path
            )
            table_store_entites = entity_extraction(lemmatized_facts)
        # table_store = generics_kb_extraction_task(generics_kb_path)

        # lemmatized_test_question = lemmatizer_task(
        #     worldtree_mapping(worldtree_test), lemmatizer_path
        # )
        # lemmatized_train_question = lemmatizer_task(
        #     worldtree_mapping(worldtree_train), lemmatizer_path
        # )

        # lemmatized_fill_facts = lemmatizer_task(
        # table_score_mapping_fill(table_store), lemmatizer_path
        # )
        # lemmatized_kindof_hyp = lemmatizer_task(
        # kind_of_mapping(table_store), lemmatizer_path
        # )
        retriever = build_search_index(lemmatized_facts)
        dev_results = bm25_search_task(
            lemmatized_question, retriever, limit=int(opts.get("limit", LIMIT))
        )
        # dev_results = read_msgpack(prediction, LIMIT, worldtree)
        # dev_abstract_results = bm25_search_task(
        #     lemmatized_abstract_dev_questions, retriever, limit=LIMIT
        # )
        # train_results = bm25_search_task(
        #     lemmatized_train_question, retriever, limit=LIMIT
        # )
        # test_results = bm25_search_task(
        #     lemmatized_test_question, retriever, limit=LIMIT
        # )
        # table_store_results = bm25_search_task(lemmatized_facts, retriever, limit=LIMIT)
        # abstract_dev_entities = entity_extraction(lemmatized_abstract_dev_questions)
        # train_entities = entity_extraction(lemmatized_train_question)
        # test_entities = entity_extraction(lemmatized_test_question)
        # kindof_hyp_entities = entity_extraction(lemmatized_kindof_hyp)
        explanation_graph = graph_extraction(
            question_query=dev_results,
            question_entities=dev_entities,
            table_store_entities=filter_facts(table_store_entites, dev_results),
            worldtree=worldtree,
            table_store=filter_facts(table_store, dev_results),
            num_nodes=LIMIT,
            training_mode=False,
        )
        answer_explanations = answer_selection(
            results=explanation_graph,
            worldtree=worldtree,
            opts=opts,
            table_store=table_store,
            training_mode=False,
            abstract_fact_limit=3,
            num_of_nodes=LIMIT,
        )
        # write_msgpack(explanationlp_output["prediction_test"], answer_explanations)

        # explanation_evaluate(answer_explanations, worldtree_dev, table_store)
        # explanation_evaluate(
        #     expl_eval_mapping(dev_results, worldtree, LIMIT), worldtree_dev, table_store
        # )

    state = flow.run()
    logger.success(f"Dev sucess {state.result[answer_explanations]._result.value}")
    return state.result[answer_explanations]._result.value


opts = {
    "question_abstract_overlap": (0.0, 1),
    "question_grouding_overlap": (0.0, 1),
    "grounding_abstract_overlap": (0.0, 1),
    "abstract_abstract_overlap": (-1, 1),
    "grounding_grounding_overlap": (-1, 0),
    "grounding_relevance": (0, 1.0),
    "abstract_relevance": (0, 1.0),
}


kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
    noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)
)
optimizer = BayesianOptimization(
    f=explanation_construction,
    pbounds=opts,
    random_state=42,
)
# optimizer.set_gp_params(kernel=kernel)
optimizer.set_gp_params(
    kernel=RationalQuadratic(length_scale=1.0)
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
)
# load_logs(
#     optimizer, logs=[f"{checkpoint_dir}/{TASK_NAME}/answer_selection_uni_test.json"],
# )


json_logger = JSONLogger(
    path=f"{checkpoint_dir}/{TASK_NAME}/answer_selection_worldtree_dev_bm25.json"
)
optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)

optimizer.maximize(init_points=20, n_iter=200, alpha=1e-3)

print(optimizer.max)
