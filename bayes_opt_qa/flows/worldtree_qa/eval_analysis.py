import os
import string
from collections import defaultdict

import msgpack
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.io as pio
import ray
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from dynaconf import settings
from loguru import logger
from prefect import Flow, Task, task
from prefect.engine.flow_runner import FlowRunner
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel
from tqdm import tqdm

from bayes_opt_qa.flows.worldtree_qa.utils import (
    explanation_mapping,
    result_mapping,
    table_score_mapping_fill,
    table_store_mapping,
    worldtree_mapping,
)
from bayes_opt_qa.tasks.answer_selection import (
    ChoiceExplanationConstruction,
    ChoiceExplanationGraph,
)
from bayes_opt_qa.tasks.eval_tasks import ExplanationEvaluationTask
from bayes_opt_qa.tasks.explanation_construction import (
    AbstractionReplaceTask,
    EntityExtractionTask,
    LPExplanationConstructionTask,
)
from bayes_opt_qa.tasks.search.bm25 import BM25FitTask, BM25SearchTask
from bayes_opt_qa.tasks.transformers.trainer import BERTSeqTrainer
from bayes_opt_qa.tasks.utils import WorldTreeLemmatizer
from poly_nlp.tasks.datasets.worldtree.evaluation_tasks import (
    WorldTreeMAPEvaluationTask,
)
from poly_nlp.tasks.datasets.worldtree.extraction_tasks import (
    TableStoreExtractionTask,
    WorldTreeExtractionTask,
    WorldTreeVersion,
)
from poly_nlp.utils.checkpoint_handler import msgpack_checkpoint_handler
from poly_nlp.utils.result_handler import MsgPackResultHandler
from poly_nlp.utils.task_runner import DSTaskRunner

pio.orca.config.use_xvfb = True
plotly.io.orca.config.executable = (
    "/home/mokanarangan/anaconda3/envs/ai-systems/bin/orca"
)

# logger.add("answer_selection_bm25_dev.log")

WORLDTREE_VERSION = WorldTreeVersion.WorldTree_V1
version = "v1"


# Setup the setting
dev_path = settings[f"worldtree_{version}"]["dev"]
table_store_path = settings[f"worldtree_{version}"]["table_store_path"]
bert_output = settings[f"worldtree_{version}"]["bert"]["bert_output"]
explanationlp_output = settings[f"worldtree_{version}"]["bert"]["explanationlp"]
checkpoint_dir = settings["checkpoint_dir"]
lemmatizer_path = settings[f"worldtree_{version}"]["lemmatizer_path"]


# Setup result handlers

TASK_NAME = f"worldtree_{WORLDTREE_VERSION.value}_eval_analysis"
default_result_handler = MsgPackResultHandler(path=checkpoint_dir, task_name=TASK_NAME)
entity_extraction = EntityExtractionTask(result_handler=default_result_handler)


# Initate the tasks
worldtree_extraction = WorldTreeExtractionTask(result_handler=default_result_handler)
table_store_extraction_task = TableStoreExtractionTask(
    result_handler=default_result_handler
)
lemmatizer_task = WorldTreeLemmatizer(result_handler=default_result_handler)


def exp_len_calculate(q_exp, table_store, entites):
    len_map = {i: i for i in range(1, 18)}
    l = len(
        [
            id
            for id in q_exp["explanation"]
            # if table_store[id]["table_name"] not in ["KINDOF", "SYNONYMY"]
        ]
    )
    text = [f"<={val}" for val in len_map]
    updated_keys = []
    for key, val in len_map.items():
        if l < val:
            updated_keys.append(key)
    return updated_keys, text, list(len_map.keys())


def concept_count(q_exp, table_store, entities):
    len_map = {i: i for i in range(1, 10)}
    l = len(
        set(
            [
                table_store[id]["table_name"]
                for id in q_exp["explanation"]
                if table_store[id]["table_name"] not in ["KINDOF", "SYNONYMY"]
            ]
        )
    )
    text = [f"<={val}" for val in len_map]
    updated_keys = []
    for key, val in len_map.items():
        if l < val:
            updated_keys.append(key)
    return updated_keys, text, list(len_map.keys())


def choice_overlap_calc(q_exp, table_store, entities):
    len_map = np.arange(1, 7, 1)
    o_score = 0
    answer_entities = set(entities[f'{q_exp["id"]}|{q_exp["answer"]}'])
    text = [f"<={val}" for val in len_map]
    for choice in q_exp["choices"].values():
        # for choice1 in q_exp["choices"].values():
        if choice != q_exp["answer"]:
            choice_entities = set(entities[f'{q_exp["id"]}|{choice}'])
            # choice_entities1 = set(entities[f'{q_exp["id"]}|{choice1}'])
            o_score = max(o_score, len(answer_entities.intersection(choice_entities)))
    updated_keys = []
    for val in len_map:
        if o_score <= val:
            updated_keys.append(val)
    return updated_keys, text, len_map


def question_len_analysis(q_exp, table_store, entities):
    len_map = np.arange(1, 15, 1)
    text = [f"<={val}" for val in len_map]

    q_len = len(set(entities[q_exp["id"]]))

    updated_keys = []
    for val in len_map:
        if q_len <= val:
            updated_keys.append(val)
    # updated_keys.append(10)
    # text.append(">=10")
    # len_map = len_map.tolist()
    # len_map.append(10)
    return updated_keys, text, len_map


@task
def bert_output_eval(bert_output, worldtree):
    with open(bert_output, "rb") as f:
        output = msgpack.unpackb(f.read(), raw=False)
    choices = {}
    scores = {}
    for key, score in output["dev"]["id_mappings"].items():
        q_id, choice = key.split("|")
        choice = worldtree[q_id]["choices"][choice]
        if q_id not in choices or scores[q_id] < score:
            choices[q_id] = choice
            scores[q_id] = score

    return choices


@task
def explantion_lp_output_eval(bert_output, worldtree):
    with open(bert_output, "rb") as f:
        output = msgpack.unpackb(f.read(), raw=False)
    choices = {}
    scores = {}
    for key, score in output.items():
        q_id, choice = key.split("|")
        if q_id not in choices or scores[q_id] < score:
            choices[q_id] = choice
            scores[q_id] = score
    return choices


Y_VAL = "Number of explanations"


@task
def plot_mapping(choices, worldtree, table_store, entities):
    total = 0
    correct = 0
    total_map = defaultdict(lambda: 0)
    correct_map = defaultdict(lambda: 0)
    for q_id, q_exp in worldtree.items():
        if q_exp["answer"] in q_exp["choices"].values():
            total += 1
            val_keys, text, len_map = exp_len_calculate(q_exp, table_store, entities)
            for k in val_keys:
                total_map[k] += 1
            if choices[q_id] == q_exp["answer"]:
                for k in val_keys:
                    correct_map[k] += 1
                correct += 1
    accuracy_map = {}
    for l, total in total_map.items():
        accuracy_map[l] = correct_map[l] / total
    exps = {k: v for k, v in sorted(accuracy_map.items(), key=lambda item: item[0])}
    return pd.DataFrame(exps.items(), columns=[Y_VAL, "Accuracy"]), text, len_map


@task
def plot_graph(b_val, e_val):
    b_out, _, _ = b_val
    e_out, text, len_map = e_val

    b_out["model"] = "BERT + UR"
    e_out["model"] = "ExplanationLP + UR"
    df = pd.concat((b_out, e_out))
    print(df)
    fig = px.line(df, x=Y_VAL, y="Accuracy", color="model", range_x=[0, 18])

    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis=dict(tickvals=len_map, ticktext=text))
    fig.write_html("output.html")
    fig.write_image("fig1.png")


@task
def worldtree_answer_mapping(worldtree, is_answer=False):
    return {
        f"{id}": f"{question_exp['question']} {question_exp['answer']}"
        if not is_answer
        else f"{question_exp['answer']}"
        for id, question_exp in worldtree.items()
    }


@task
def worldtree_choice_mapping(worldtree):
    return {
        f"{id}|{choice}": f"{choice}"
        for id, question_exp in worldtree.items()
        for choice in question_exp["choices"].values()
    }


def explanation_construction(**opts):
    # Build the flow
    with Flow("Question answering with WorldTreeCorpus") as flow:

        worldtree_dev = worldtree_extraction(
            dev_path,
            WORLDTREE_VERSION,
            skip_missing_explanations=True,
            skip_empty_explanations=False,
        )
        table_store = table_store_extraction_task(table_store_path, original_map=True)
        lemmatized_question = lemmatizer_task(
            worldtree_answer_mapping(worldtree_dev, is_answer=False), lemmatizer_path
        )
        lemmatized_answer = lemmatizer_task(
            worldtree_choice_mapping(worldtree_dev), lemmatizer_path
        )
        question_entities = entity_extraction(lemmatized_question)
        answer_entities = entity_extraction(lemmatized_answer)
        b_out = plot_mapping(
            bert_output_eval(bert_output, worldtree_dev),
            worldtree_dev,
            table_store,
            question_entities,
        )
        e_out = plot_mapping(
            explantion_lp_output_eval(explanationlp_output, worldtree_dev),
            worldtree_dev,
            table_store,
            question_entities,
        )
        plot_graph(b_out, e_out)

    FlowRunner(flow=flow, task_runner_cls=DSTaskRunner).run(
        task_runner_state_handlers=[msgpack_checkpoint_handler]
    )


uni_dev_answer_opts = {
    "abstraction_overlap": -0.8851856387522329,
    # "abstraction_overlap": -1,
    "abstraction_question_overlap": 1.3272362748898874e-11,
    "abstraction_relevance": 0.5788896361127343,
    "challenge_explanations": 2.8281604737671957,
    # "challenge_explanations": 2,
    "unification_abstraction_overlap": 0.0,
    # "unification_abstraction_overlap": 0.1,
    "unification_overlap": 0.0829835403843482,
    "unification_question_overlap": 0.4496807212823613,
    "unification_relevance": 0.8605042917116489,
    # "unification_relevance": 0.955042917116489,
    "wrong_overlap": -0.06933494548515952,
}


explanation_construction(**uni_dev_answer_opts)
