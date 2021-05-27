import json
import math
from collections import defaultdict
from functools import partial, reduce
from typing import Dict

import numpy as np
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm

FOLD = "fold"
CHALLENGE = "Challenge"
EASY = "Easy"


def qa_evaluation(
    dataset, indexes: np.array, preds: np.array, output_file=None, **kwargs
):
    preds = preds[:, 1]
    q_ids = list(map(lambda index: dataset.get_id(index), indexes))
    scores, choices = defaultdict(lambda: -math.inf), defaultdict(lambda: None)
    for index, (id, choice) in enumerate(q_ids):
        if preds[index] > scores[id]:
            scores[id] = preds[index]
            choices[id] = choice
    total, correct = 0, 0
    t_easy, c_easy = 0, 0
    t_challenge, c_challenge = 0, 0
    output_map = {}
    for id, q_exp in tqdm(dataset.world_tree.items(), "Calculating values"):
        if q_exp["answerKey"] in q_exp["choices"]:
            total += 1
            if q_exp[FOLD] == CHALLENGE:
                t_challenge += 1
            elif q_exp[FOLD] == EASY:
                t_easy += 1
            if q_exp["answerKey"] == choices[id]:
                output_map[id] = True
                if q_exp[FOLD] == CHALLENGE:
                    c_challenge += 1
                elif q_exp[FOLD] == EASY:
                    c_easy += 1
                correct += 1
            else:
                output_map[id] = False
    if output_file is not None:
        logger.info("Output file provided. Saving output map to file")
        with open(output_file, "w") as fp:
            json.dump(output_map, fp)
    logger.info(f"Correct: {correct}, Total: {total}")
    acc = correct / total
    e_acc = c_easy / t_easy if t_easy > 0 else 0.0
    c_acc = c_challenge / t_challenge if t_challenge > 0 else 0.0
    logger.info(f"Easy acc: {e_acc}, Challenge acc: {c_acc}")
    return acc
