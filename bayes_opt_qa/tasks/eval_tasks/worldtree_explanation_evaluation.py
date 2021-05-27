from collections import defaultdict

from loguru import logger
from overrides import overrides
from prefect import Task
from sklearn.metrics import precision_recall_fscore_support


class ExplanationEvaluationTask(Task):
    def run(self, results, world_tree, table_store):
        correct_count = 0
        output_map = results[1]
        case_count = defaultdict(lambda: 0)
        explanations = results[2]
        count = 0
        m_precision = 0
        m_recall = 0
        m_f1 = 0
        for key, r_expl in explanations.items():
            q_id, choice = key.split("|")
            if choice != world_tree[q_id]["answer"]:
                continue
            count += 1
            true_exp = set(
                [
                    id
                    for id in world_tree[q_id]["explanation"].keys()
                    if id in table_store and table_store[id]["type"] == "ABSTRACT"
                ]
            )
            if len(true_exp) == 0:
                m_precision += 1
                m_recall += 1
                m_f1 += 1

            pred = set(
                [
                    id
                    for index, id in enumerate(r_expl)
                    if table_store[id]["type"] == "ABSTRACT"
                    # if index < abstract_limit
                ]
            )
            precision = (
                len(true_exp.intersection(pred)) / len(pred) if len(pred) > 0 else 0.0
            )
            to_print = False
            if output_map[q_id]:
                correct_count += 1
                if abs(precision - 1) < 0.001:
                    case_count[0] += 1
                elif abs(precision - 0) < 0.001:
                    to_print = True
                    case_count[1] += 1
                else:
                    case_count[2] += 1
            if to_print:
                q_exp = world_tree[q_id]
                print(f'Question: {q_exp["question"]}')
                print(f'Answert: {q_exp["answer"]}')
                print(q_exp["choices"].values())
                print("------------------------------")
                for e_id in explanations[f'{q_id}|{q_exp["answer"]}']:
                    print(
                        table_store[e_id]["fact"],
                        "-",
                        table_store[e_id]["type"],
                    )
                # print("*****Gold Explanations*************")
                # for e_id in q_exp["explanation"]:
                #     if e_id in table_store:
                #         print(
                #             table_store[e_id]["fact"],
                #             "-",
                #             table_store[e_id]["type"],
                #         )
                print("******************************")

            recall = (
                len(true_exp.intersection(pred)) / len(true_exp)
                if len(true_exp) > 0
                else 1.0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            m_precision += precision
            m_recall += recall
            m_f1 += f1
        m_f1 = m_f1 / count
        m_precision = m_precision / count
        m_recall = m_recall / count

        for case, c_count in case_count.items():
            logger.success(f"Case {case}: {c_count/correct_count}")

        logger.success(f"f1 score: {m_f1}")
        logger.success(f"Precision score: {m_precision}")
        logger.success(f"Recall score: {m_recall}")
        return m_f1
        # predicted =
