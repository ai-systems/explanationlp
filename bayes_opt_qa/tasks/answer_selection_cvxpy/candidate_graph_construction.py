import heapq
import itertools
import math
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import ray
import scipy as sp
from loguru import logger
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from prefect import Task
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


class CandidateGraphConstruction(Task):
    @staticmethod
    def calc_interoverlap(e1: List, e2: List):
        e1 = set(e1)
        e2 = set(e2)
        score = (
            len(e1.intersection(e2)) / max(len(e1), len(e2))
            if max(len(e1), len(e2)) > 0
            else 0.0
        )
        return score, e1.intersection(e2)

    @staticmethod
    def calc_interoverlap2(e1: List, e2: List):
        e1 = set(e1)
        e2 = set(e2)
        score = len(e1.intersection(e2)) / len(e1) if len(e1) > 0 else 0.0
        return score, e1.intersection(e2)

    @staticmethod
    def calc_interoverlap_question(e1, e2):
        score = (
            len(e1.intersection(e2)) / max(len(e1), len(e2))
            if max(len(e1), len(e2)) > 0
            else 0.0
        )
        return score

    @staticmethod
    def construct_graphs(
        pos,
        input,
        table_store,
        table_store_entities,
        question_enitities,
        num_nodes,
        # question_grounding,
        question_abstract_facts,
        grounding_limit,
        table_store_embeddings,
        question_embedding,
        abstract_grounding_limit,
        entity_mappings,
        question_entity_limit=15,
    ):
        NO_OF_NODES = num_nodes + 1
        explanation_graphs = {}
        for q_id, question in tqdm(input.items()):
            # for q_id, question in input.items():
            inference_chains = []
            grounding_nodes = {}
            abstract_nodes = {}
            grounding_abstract_overlap_maps = []
            grounding_lexical_nodes = {}
            abstract_lexical_nodes = {}
            q_ovrlp_trms = {}
            for a_id, score in question_abstract_facts[q_id].items():
                if (
                    len(abstract_nodes)
                    > num_nodes - grounding_limit - abstract_grounding_limit
                ):
                    break
                (
                    overlap_score,
                    overlap_terms,
                ) = CandidateGraphConstruction.calc_interoverlap(
                    question_enitities[q_id], table_store_entities[a_id]
                )
                if True:
                    # if score > 0.0 and overlap_score > 0.0:
                    q_ovrlp_trms[a_id] = overlap_terms

                    # if True:
                    inference_chains.append(
                        (
                            a_id,
                            overlap_score,
                            set(question_enitities[q_id]).intersection(
                                set(table_store_entities[a_id])
                            ),
                            "abstract",
                        )
                    )
                    abstract_nodes[a_id] = 1 - cosine(
                        question_embedding[q_id], table_store_embeddings[a_id]
                    )
                    abstract_lexical_nodes[a_id] = score

            max_grounding_node_count = (
                num_nodes - len(abstract_nodes) - abstract_grounding_limit
            )
            # if q_id in question_grounding:
            # for g_id, g_score in question_grounding[q_id].items():
            for g_id, fact in table_store.items():
                if fact["type"] == "ABSTRACT":
                    continue

                g_score = 1
                if len(grounding_nodes) > max_grounding_node_count - 1:
                    break
                (
                    overlap_score,
                    overlap_terms,
                ) = CandidateGraphConstruction.calc_interoverlap2(
                    table_store_entities[g_id], question_enitities[q_id]
                )
                if g_score > 0 and overlap_score > 0:
                    has_abstract_overlap = False
                    for a_id in abstract_nodes:
                        (
                            a_o_score,
                            a_o_terms,
                        ) = CandidateGraphConstruction.calc_interoverlap(
                            table_store_entities[g_id], table_store_entities[a_id]
                        )
                        if (
                            len(
                                set(["use", "part", "made", "locate"]).intersection(
                                    a_o_terms
                                )
                            )
                            > 0
                        ):
                            continue
                        if (
                            a_o_score > 0
                            and not overlap_terms == a_o_terms
                            and not len(overlap_terms.intersection(a_o_terms)) > 0
                        ):
                            grounding_abstract_overlap_maps.append((g_id, a_id))
                            # print(question)
                            # print(table_store[a_id]["fact"])
                            # print(table_store[g_id]["fact"])
                            # print(
                            #     overlap_terms,
                            #     "->",
                            #     a_o_terms,
                            # )
                            # print("--------------------------")
                            has_abstract_overlap = True
                    if not has_abstract_overlap:
                        continue
                    q_ovrlp_trms[g_id] = overlap_terms
                    limited_overlaps = True
                    # for t in overlap_terms:
                    #     q_ground_overlaps[t] += 1
                    #     if q_ground_overlaps[t] > 3:
                    #         limited_overlaps = False
                    # if not limited_overlaps:
                    #     continue
                    # print(overlap_terms)
                    # print(table_store_entities[g_id])
                    # print(table_store_entities[a_id])
                    # print("-------------------------")

                    # grounding_nodes[g_id] = 1 - cosine(
                    #     question_embedding[q_id], table_store_embeddings[g_id]
                    # )
                    grounding_nodes[g_id] = 1
                    grounding_lexical_nodes[g_id] = g_score
                    inference_chains.append(
                        (
                            g_id,
                            overlap_score,
                            set(question_enitities[q_id]).intersection(
                                set(table_store_entities[g_id])
                            ),
                            "grounding",
                        )
                    )
            for a1_id in abstract_nodes:
                for en in table_store_entities[a1_id]:
                    for g_id in entity_mappings[en]:
                        if len(grounding_nodes) + len(abstract_nodes) > num_nodes:
                            break
                        (
                            a1_o_score,
                            a1_o_terms,
                        ) = CandidateGraphConstruction.calc_interoverlap2(
                            table_store_entities[g_id],
                            table_store_entities[a1_id],
                        )
                        if table_store[g_id]["type"] == "GROUNDING":
                            for a2_id in abstract_nodes:
                                if a1_id != a2_id:
                                    (
                                        a2_o_score,
                                        a2_o_terms,
                                    ) = CandidateGraphConstruction.calc_interoverlap2(
                                        table_store_entities[g_id],
                                        table_store_entities[a2_id],
                                    )
                                    if (
                                        a1_o_score > 0
                                        and a2_o_score > 0
                                        and len(a1_o_terms.intersection(a2_o_terms)) < 0
                                    ):
                                        print(table_store[a1_id]["fact"])
                                        print(table_store[a2_id]["fact"])
                                        print(table_store[g_id]["fact"])
                                        print("------------------------------")
                                        print(here)
            # print(len(grounding_nodes))
            question_abstract_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            question_grounding_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            grounding_abstract_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            abstract_abstract_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            abstract_abstract_q_overlap_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            abstract_abstract_overlap_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            grounding_grounding_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, NO_OF_NODES), dtype=np.float64
            )
            entity_fact_edges = sp.sparse.lil_matrix(
                (NO_OF_NODES, question_entity_limit), dtype=np.int
            )
            grounding_facts = np.zeros(NO_OF_NODES, dtype=np.float64)
            abstract_facts = np.zeros(NO_OF_NODES, dtype=np.float64)
            l_grounding_facts = np.zeros(NO_OF_NODES, dtype=np.float64)
            l_abstract_facts = np.zeros(NO_OF_NODES, dtype=np.float64)

            node_mapping = {
                t_id: index + 1
                for index, t_id in enumerate({**abstract_nodes, **grounding_nodes})
            }
            # print(node_mapping)
            question_entity_mapping = {
                entity: index
                for index, entity in enumerate(question_enitities)
                if index < question_entity_limit
            }
            assert len(node_mapping) + 1 <= NO_OF_NODES, f"{len(node_mapping)}"
            for id, index in node_mapping.items():
                if id in abstract_nodes:
                    abstract_facts[index] = abstract_nodes[id]
                    l_abstract_facts[index] = abstract_lexical_nodes[id]

                elif id in grounding_nodes:
                    grounding_facts[index] = grounding_nodes[id]
                    l_grounding_facts[index] = grounding_lexical_nodes[id]
            for chain in inference_chains:
                t_id, score, entity_overlaps, c_type = chain
                entity_node = t_id
                if c_type == "abstract":
                    question_abstract_edges[0, node_mapping[t_id]] = score
                elif c_type == "grounding":
                    # print("here")
                    question_grounding_edges[0, node_mapping[t_id]] = score
                # print(
                #     "Question", "->", table_store[t_id]["fact"], "-", c_type, "-", t_id
                # )
                for ent in entity_overlaps:
                    if ent in question_entity_mapping:
                        entity_fact_edges[
                            node_mapping[entity_node], question_entity_mapping[ent]
                        ] = 1
                    # else:
                    # print(f"{ent} not mapped")
            # print("---------------------------")
            for a1_id in abstract_nodes:
                for a2_id in abstract_nodes:
                    if a1_id != a2_id:
                        abstract_abstract_overlap_edges[
                            node_mapping[a1_id], node_mapping[a2_id]
                        ] = CandidateGraphConstruction.calc_interoverlap_question(
                            q_ovrlp_trms[a1_id], q_ovrlp_trms[a2_id]
                        )
                        abstract_abstract_edges[
                            node_mapping[a1_id], node_mapping[a2_id]
                        ] = 1 - cosine(
                            table_store_embeddings[a1_id], table_store_embeddings[a2_id]
                        )
            for g_id, a_id in grounding_abstract_overlap_maps:
                (
                    a_g_score,
                    overlap_terms,
                ) = CandidateGraphConstruction.calc_interoverlap2(
                    table_store_entities[g_id], table_store_entities[a_id]
                )
                grounding_abstract_edges[
                    node_mapping[g_id], node_mapping[a_id]
                ] = a_g_score

            # for g_id in grounding_nodes:
            #     has_overlap = False
            #     for a_id in abstract_nodes:
            #         (
            #             a_g_score,
            #             overlap_terms,
            #         ) = CandidateGraphConstruction.calc_interoverlap2(
            #             table_store_entities[g_id], table_store_entities[a_id]
            #         )
            #         if a_g_score > 0:
            #             has_overlap = True
            #         grounding_abstract_edges[
            #             node_mapping[g_id], node_mapping[a_id]
            #         ] = a_g_score
            #     assert has_overlap
            # TODO: Revisit this
            for g1_id in grounding_nodes:
                for g2_id in grounding_nodes:
                    if g1_id != g2_id:
                        grounding_grounding_edges[
                            node_mapping[g1_id], node_mapping[g2_id]
                        ] = CandidateGraphConstruction.calc_interoverlap_question(
                            q_ovrlp_trms[g1_id], q_ovrlp_trms[g2_id]
                        )
                    # grounding_grounding_edges[
                    #     node_mapping[g1_id], node_mapping[g2_id]
                    # ] = math.ceil(
                    #     CandidateGraphConstruction.calc_interoverlap(
                    #         table_store_entities[g1_id], table_store_entities[g2_id]
                    #     )[0]
                    # )
                    # print(q_ovrlp_trms[g1_id], q_ovrlp_trms[g2_id], "grounding")
            explanation_graphs[q_id] = {}
            explanation_graphs[q_id][
                "question_abstract_edges"
            ] = question_abstract_edges
            explanation_graphs[q_id][
                "question_grounding_edges"
            ] = question_grounding_edges
            explanation_graphs[q_id][
                "grounding_abstract_edges"
            ] = grounding_abstract_edges
            explanation_graphs[q_id][
                "abstract_abstract_edges"
            ] = abstract_abstract_edges
            explanation_graphs[q_id][
                "abstract_abstract_overlap_edges"
            ] = abstract_abstract_overlap_edges
            explanation_graphs[q_id][
                "grounding_grounding_edges"
            ] = grounding_grounding_edges
            explanation_graphs[q_id]["grounding_facts"] = grounding_facts
            explanation_graphs[q_id]["abstract_facts"] = abstract_facts
            explanation_graphs[q_id]["lexical_grounding_facts"] = l_grounding_facts
            explanation_graphs[q_id]["lexical_abstract_facts"] = l_abstract_facts
            explanation_graphs[q_id]["entity_fact_edges"] = entity_fact_edges
            explanation_graphs[q_id]["node_mapping"] = {
                val: key for key, val in node_mapping.items()
            }
        return explanation_graphs

    @overrides
    def run(
        self,
        lemmatized_questions,
        table_store,
        table_store_entities,
        question_enitities,
        question_abstract_facts,
        question_grounding,
        grounding_limit,
        num_nodes,
        table_store_embeddings,
        question_embedding,
        abstract_grounding_limit,
        entity_mappings,
        **kwargs,
    ):

        ray_executor = RayExecutor()
        explanation_graphs = ray_executor.run(
            lemmatized_questions,
            self.construct_graphs,
            dict(
                table_store=table_store,
                table_store_entities=table_store_entities,
                question_enitities=question_enitities,
                # question_grounding=question_grounding,
                question_abstract_facts=question_abstract_facts,
                grounding_limit=grounding_limit,
                num_nodes=num_nodes,
                table_store_embeddings=table_store_embeddings,
                question_embedding=question_embedding,
                abstract_grounding_limit=abstract_grounding_limit,
                entity_mappings=entity_mappings,
            ),
            # batch_count=8,
            **kwargs,
        )

        return explanation_graphs
