import math
import multiprocessing
import os
import time
from functools import reduce
from typing import Dict, List

import cvxpy as cp
import numpy as np
import ray
import scipy as sp
import torch
from bayes_opt_qa.evaluation.qa_evaluation_ilp import qa_evaluation
from loguru import logger
from nltk.util import ngrams
from overrides import overrides
from poly_nlp.utils.data_manipulation.data_manipulation import create_dict_chunks
from prefect import Task
from tqdm import tqdm


class AnswerSelectionCVXPY(Task):
    # @staticmethod
    @ray.remote
    def solve_problem(
        pos,
        results,
        opts,
        num_of_nodes,
        abstract_fact_limit,
        thread_count,
        question_entity_limit=15,
    ):
        NO_OF_NODES = num_of_nodes + 1
        nodes = cp.Variable((NO_OF_NODES), boolean=True)
        edges = cp.Variable((NO_OF_NODES, NO_OF_NODES), boolean=True)
        # coverage = cp.Variable((question_entity_limit), boolean=True)

        node_weights = cp.Parameter((NO_OF_NODES))
        inter_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES))
        incoming_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES))
        edge_weights_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES))
        entity_edges_param = cp.Parameter((NO_OF_NODES, question_entity_limit))
        is_abstract_fact_param = cp.Parameter((NO_OF_NODES))

        objective = cp.Maximize(
            opts.get("c_node_wi", 1) * cp.sum(cp.multiply(node_weights, nodes))
            + opts.get("c_edge_wi", 1)
            * cp.sum(
                cp.reshape(
                    cp.multiply(edge_weights_param, edges), (NO_OF_NODES * NO_OF_NODES)
                )
            )
            # + opts.get("c_coverage_wi", 1) * cp.sum(coverage)
        )
        constraints = [
            # nodes <= 1,
            # edges <= 1,
            # coverage <= 1,
            nodes[0] == 1,
            nodes - incoming_edges_param @ nodes <= 0,
            cp.multiply(
                inter_edges_param,
                edges - np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - cp.transpose(
                    (np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)))
                ),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - (
                    np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES))
                    + cp.transpose(
                        (
                            np.ones((NO_OF_NODES, 1))
                            @ cp.reshape(nodes, (1, NO_OF_NODES))
                        )
                    )
                    - np.ones((NO_OF_NODES, NO_OF_NODES))
                ),
            )
            >= 0,
            cp.sum(cp.multiply(is_abstract_fact_param, nodes)) <= abstract_fact_limit,
            # cp.sum(cp.multiply(is_abstract_fact_param, nodes)) >= 2,
            (incoming_edges_param + incoming_edges_param.T) @ nodes
            - 2
            + 2 * (1 - cp.multiply(nodes, 1 - is_abstract_fact_param))
            >= 0,
            # coverage <= cp.transpose(entity_edges_param) @ nodes,
        ]
        prob = cp.Problem(objective, constraints)

        graph_score = {}
        explanations = {}
        results: Dict[str, sp.sparse.csr_matrix]
        for key, results in results.items():
            # for key, results in tqdm(results.items(), position=pos):
            q_id, choice = key.split("|")

            question_abstract_edges = results["question_abstract_edges"].toarray()
            question_grounding_edges = results["question_grounding_edges"].toarray()
            grounding_abstract_edges = results["grounding_abstract_edges"].toarray()
            entity_edges = results["entity_fact_edges"].toarray()

            outgoing_edges_weigths = (
                question_abstract_edges
                + question_grounding_edges
                + grounding_abstract_edges
            )
            outgoing_edges_weigths[0][0] = 1
            abstract_abstract_overlap_edges = results[
                "abstract_abstract_overlap_edges"
            ].toarray()
            abstract_abstract_edges = results["abstract_abstract_edges"].toarray()
            grounding_grounding_edges = results["grounding_grounding_edges"].toarray()
            inter_edges_weights = (
                abstract_abstract_overlap_edges
                + grounding_grounding_edges
                + abstract_abstract_edges
            )

            # grounding_facts = (
            #     opts["grounding_relevance"] * results["grounding_facts"]
            #     + opts["g_lexical_relevance"] * results["lexical_grounding_facts"]
            # ) / 2
            # grounding_facts = opts["grounding_relevance"] * results["grounding_facts"]
            abstract_facts = (
                opts["abstract_relevance"] * results["abstract_facts"]
                + opts["a_lexical_relevance"] * results["lexical_abstract_facts"]
            ) / 2
            # abstract_facts = results["abstract_facts"]

            is_abstract_fact = np.where(
                abstract_facts != 0,
                np.ones_like(abstract_facts),
                np.zeros_like(abstract_facts),
            )
            # is_abstract_fact[0] = 1
            incoming_edges = np.where(
                outgoing_edges_weigths.T != 0,
                np.ones_like(outgoing_edges_weigths),
                np.zeros_like(outgoing_edges_weigths),
            )
            inter_edges = np.where(
                inter_edges_weights != 0,
                np.ones_like(inter_edges_weights),
                np.zeros_like(inter_edges_weights),
            )

            # abstract_abstract_edges = np.where(
            #     abstract_abstract_edges > opts.get("abstract_relevance_limit"),
            #     np.zeros_like(abstract_abstract_edges),
            #     abstract_abstract_edges,
            # )
            edge_weights = (
                np.triu(
                    opts["abstract_abstract_overlap"] * abstract_abstract_overlap_edges
                    + opts.get("abstract_abstract_s_overlap") * abstract_abstract_edges
                    + opts["grounding_grounding_overlap"] * grounding_grounding_edges
                )
                + (
                    opts["question_abstract_overlap"] * question_abstract_edges
                    + opts["question_grouding_overlap"] * question_grounding_edges
                    + opts["grounding_abstract_overlap"] * grounding_abstract_edges
                ).T
            )

            # negative_edges = np.where(
            #     grounding_grounding_edges != 0,
            #     -1 * np.ones_like(grounding_grounding_edges),
            #     np.ones_like(grounding_grounding_edges),
            # )

            # edge_weights = torch.nn.functional.softmax(
            #     torch.tensor(edge_weights), dim=-1
            # ).numpy()
            # node_weights_normalized = torch.nn.functional.softmax(
            #     torch.tensor(abstract_facts + grounding_facts), dim=0
            # ).numpy()

            # edge_weights = edge_weights * negative_edges
            is_abstract_fact_param.value = is_abstract_fact
            incoming_edges_param.value = incoming_edges
            inter_edges_param.value = np.triu(inter_edges) + incoming_edges
            edge_weights_param.value = edge_weights
            # node_weights.value = node_weights_normalized * 100
            node_weights.value = abstract_facts
            # node_weights.value = abstract_facts + grounding_facts
            entity_edges_param.value = entity_edges

            # Solving
            result = prob.solve(
                solver=cp.CPLEX,
                # verbose=True,
                cplex_params={"threads": thread_count, "timelimit": 5},
                # cplex_params={"threads": thread_count},
            )

            if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
                print("Optimal solution not achieved. Error")
                # logger.error(f"{q_id} not optimal solution. Output: {prob.status}")
                graph_score[f"{q_id}|{choice}"] = -math.inf
                explanations[f"{q_id}|{choice}"] = []
            else:
                if prob.status == cp.OPTIMAL_INACCURATE:
                    print("Optimal solution not achieved")
                graph_score[f"{q_id}|{choice}"] = result
                explanations[f"{q_id}|{choice}"] = []
                for index, node_val in enumerate(nodes.value):
                    if abs(node_val - 1) < 0.0001 and index != 0:
                        explanations[f"{q_id}|{choice}"].append(
                            results["node_mapping"][index]
                        )
        return graph_score, explanations

    @overrides
    def run(
        self,
        results,
        worldtree,
        num_of_nodes,
        abstract_fact_limit,
        opts={},
        thread_count=3,
        training_mode=True,
        output_path=None,
    ):
        logger.info(f"Opts : {opts}")
        os.environ["OMP_NUM_THREADS"] = f"{thread_count}"

        batch_count = math.floor(multiprocessing.cpu_count() / thread_count)
        batch_size = math.ceil(len(results) / batch_count)

        logger.info(f"Batch_size = {batch_size}")
        logger.info(f"Number of batches = {batch_count}")
        logger.info(f"Worldtree instances: {len(worldtree)}")

        batches = create_dict_chunks(results, batch_size)

        logger.info("Optimizing graphs")
        start = time.time()

        batch_results = ray.get(
            [
                self.solve_problem.remote(
                    pos=pos,
                    results=batch,
                    opts=opts,
                    num_of_nodes=num_of_nodes,
                    abstract_fact_limit=abstract_fact_limit,
                    thread_count=thread_count,
                    # wordltree=worldtree,
                )
                for pos, batch in enumerate(tqdm(batches))
            ]
        )
        graph_score = reduce(lambda x, y: {**x, **y[0]}, batch_results, {})
        explanations = reduce(lambda x, y: {**x, **y[1]}, batch_results, {})
        # graph_score = self.solve_problem(
        #     pos=0,
        #     results=results,
        #     opts=opts,
        #     num_of_nodes=num_of_nodes,
        #     abstract_fact_limit=abstract_fact_limit,
        # )
        end = time.time()
        logger.success(f"Optimizing time: {end-start}")
        score, output_map = qa_evaluation(
            worldtree, graph_score, output_file="dev.json"
        )

        return score, output_map, explanations, graph_score
        # return selected_explanations
