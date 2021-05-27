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


def qa_evaluation(worldtree, preds, output_file=None):
    scores, choices = defaultdict(lambda: -math.inf), defaultdict(lambda: None)
    for key, ilp_score in preds.items():
        id, choice = key.split("|")
        # print(choice, ilp_score, id)
        if ilp_score > scores[id]:
            scores[id] = ilp_score
            choices[id] = choice
    output_map = {}
    total, correct = 0, 0
    t_easy, c_easy = 0, 0
    t_challenge, c_challenge = 0, 0
    for id, q_exp in tqdm(worldtree.items(), "Calculating values"):
        if q_exp["answer"] in q_exp["choices"].values():
            total += 1
            if q_exp[FOLD] == CHALLENGE:
                t_challenge += 1
            elif q_exp[FOLD] == EASY:
                t_easy += 1
            if q_exp["answer"] == choices[id]:
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
    logger.info(f"Correct Challenge: {c_challenge}, Total Challenge: {t_challenge}")
    acc = correct / total
    e_acc = c_easy / t_easy if t_easy > 0 else 0.0
    c_acc = c_challenge / t_challenge if t_challenge > 0 else 0.0
    logger.info(f"Easy acc: {e_acc}, Challenge acc: {c_acc}")
    logger.success(f"Total: {acc}")
    # return (e_acc + c_acc) / 2
    return acc, output_map
