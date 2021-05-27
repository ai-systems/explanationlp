import math
import multiprocessing
import time
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Set

import numpy as np
import ray
import scipy as sp
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm

from bayes_opt_qa.tasks.explanation_construction.entity_extraction import (
    EntityExtractionTask,
)
from poly_nlp.utils.data_manipulation.data_manipulation import create_dict_chunks


class GraphConstructionCVXPY(Task):
    @staticmethod
    def calc_interoverlap(e1: List, e2: List):
        e1 = set(e1)
        e2 = set(e2)
        score = len(e1.intersection(e2)) / max(len(e1), len(e2))
        div_score = (max(len(e1), len(e2)) - len(e1.intersection(e2))) / max(
            len(e1), len(e2)
        )
        return score, div_score

    @staticmethod
    def calc_interoverlap_target(source: List, target: List):
        source = set(source)
        target = set(target)
        score = len(source.intersection(target)) / len(source) if len(source) > 0 else 0
        div_score = len(target - source) / len(source) if len(source) > 0 else 0

        return score, div_score

    @ray.remote
    def linear_programming(
        pos,
        worldtree,
        grounding_entities,
        abstract_entities,
        question_query,
        table_store_entities,
        question_entities,
        table_store,
        num_nodes=30,
        training_mode=True,
    ):
        NO_OF_NODES = num_nodes + 1
        explanation_graphs = {}
        for key, results in question_query.items():
            q_id, choice = key.split("|")
            # print(worldtree[q_id]["question"])
            # print(choice)
            # print()
            # for id in results:
            #     print(table_store[id]["fact"] + " " + table_store[id]["type"])
            # print("----------------------------")
            if training_mode == True and choice != worldtree[q_id]["answer"]:
                continue
            question_abstract_edges = sp.sparse.csr_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            question_grounding_edges = sp.sparse.csr_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            grounding_abstract_edges = sp.sparse.csr_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            abstract_abstract_edges = sp.sparse.csr_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            grounding_grounding_edges = sp.sparse.csr_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            grounding_facts = np.zeros(NO_OF_NODES, dtype=np.float64)
            abstract_facts = np.zeros(NO_OF_NODES, dtype=np.float64)

            node_mapping = {t_id: index + 1 for index, t_id in enumerate(results)}
            assert len(node_mapping) + 1 == NO_OF_NODES

            for t_id, score in results.items():
                (
                    overlap_score,
                    div_score,
                ) = GraphConstructionCVXPY.calc_interoverlap_target(
                    table_store_entities[t_id], question_entities[key]
                )
                if overlap_score > 0:
                    if t_id in abstract_entities:
                        abstract_facts[node_mapping[t_id]] = score
                        question_abstract_edges[0, node_mapping[t_id]] = overlap_score
                    elif t_id in grounding_entities:
                        grounding_facts[node_mapping[t_id]] = score
                        question_grounding_edges[0, node_mapping[t_id]] = overlap_score
            for t1_id, index1 in node_mapping.items():
                for t2_id, index2 in node_mapping.items():
                    (
                        overlap_score,
                        div_score,
                    ) = GraphConstructionCVXPY.calc_interoverlap(
                        table_store_entities[t1_id], table_store_entities[t2_id]
                    )
                    if overlap_score > 0:
                        if t1_id in abstract_entities and t2_id in abstract_entities:
                            abstract_abstract_edges[index1, index2] = overlap_score
                        elif t1_id in grounding_entities and t2_id in abstract_entities:
                            grounding_abstract_edges[index1, index2] = overlap_score
                        elif (
                            t1_id in grounding_entities and t2_id in grounding_entities
                        ):
                            grounding_grounding_edges[index1, index2] = overlap_score
            explanation_graphs[key] = {}
            explanation_graphs[key]["question_abstract_edges"] = question_abstract_edges
            explanation_graphs[key][
                "question_grounding_edges"
            ] = question_grounding_edges
            explanation_graphs[key][
                "grounding_abstract_edges"
            ] = grounding_abstract_edges
            explanation_graphs[key]["abstract_abstract_edges"] = abstract_abstract_edges
            explanation_graphs[key][
                "grounding_grounding_edges"
            ] = grounding_grounding_edges
            explanation_graphs[key]["grounding_facts"] = grounding_facts
            explanation_graphs[key]["abstract_facts"] = abstract_facts
        return explanation_graphs

    @overrides
    def run(
        self,
        question_query,
        question_entities,
        table_store_entities,
        worldtree,
        table_store,
        num_nodes,
        training_mode=True,
    ):
        grounding_entities = {
            id: set(table_store_entities[id])
            for id, fact in table_store.items()
            # if fact["table_name"] in ["KINDOF", "SYNONYMY"]
            if fact["type"] == "GROUNDING"
        }
        abstract_entities = {
            id: set(table_store_entities[id])
            for id, fact in table_store.items()
            # if fact["table_name"] not in ["KINDOF", "SYNONYMY"]
            if fact["type"] == "ABSTRACT"
        }
        assert len(grounding_entities) > 0
        assert len(abstract_entities) > 0

        batch_count = multiprocessing.cpu_count()
        batch_size = math.ceil(len(question_query) / batch_count)

        logger.info(f"Batch_size = {batch_size}")
        logger.info(f"Number of batches = {batch_count}")
        logger.info(f"Worldtree instances: {len(worldtree)}")

        batches = create_dict_chunks(question_query, batch_size)

        logger.info("Building graphs")
        start = time.time()

        batch_results = ray.get(
            [
                self.linear_programming.remote(
                    pos=pos,
                    worldtree=worldtree,
                    grounding_entities=grounding_entities,
                    abstract_entities=abstract_entities,
                    question_query=batch,
                    table_store_entities=table_store_entities,
                    question_entities=question_entities,
                    table_store=table_store,
                    num_nodes=num_nodes,
                    training_mode=training_mode,
                )
                for pos, batch in enumerate(tqdm(batches))
            ]
        )

        explanation_graphs = reduce(lambda x, y: {**x, **y}, batch_results, {})
        end = time.time()
        logger.success(f"Graph building success. Time taken: {end-start}")

        return explanation_graphs
