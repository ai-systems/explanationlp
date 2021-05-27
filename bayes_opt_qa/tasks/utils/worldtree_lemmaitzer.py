import math
import multiprocessing
import time
from functools import reduce
from typing import Dict

import nltk
import ray
import spacy
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from overrides import overrides
from prefect import Task
from tqdm import tqdm

from poly_nlp.utils.data_manipulation.data_manipulation import create_dict_chunks

stop = stopwords.words("english")


class WorldTreeLemmatizer(Task):
    @staticmethod
    def lemma(query, lemmatizer, lemmas={}):
        temp = []
        query = (
            query.replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
        )
        return " ".join(
            [token.lemma_ for token in lemmatizer(query) if token.lemma_ not in stop]
        )

    @staticmethod
    def query(query_dict, lemmas, stop):
        nlp = spacy.load("en_core_web_sm")
        query_output = {}
        for id, query in query_dict.items():
            query_output[id] = WorldTreeLemmatizer.lemma(query, nlp, {})
        return query_output

    @overrides
    def run(self, query_dict: Dict, lematizer_path: str, no_parallel=False):
        batch_count = multiprocessing.cpu_count()
        batch_size = math.ceil(len(query_dict) / batch_count)

        logger.info(f"Batch_size = {batch_size}")
        logger.info(f"Number of batches = {batch_count}")

        batches = create_dict_chunks(query_dict, batch_size)

        logger.info("Lemmatizing")
        start = time.time()

        if not no_parallel:
            remote_fn = ray.remote(self.query)
            batch_results = ray.get(
                [
                    remote_fn.remote(query_dict=batch, lemmas={}, stop=stop)
                    for pos, batch in enumerate(tqdm(batches))
                ]
            )
            query_output = reduce(lambda x, y: {**x, **y}, batch_results, {})
        else:
            query_output = self.query(query_dict=query_dict, lemmas={}, stop=stop)

        end = time.time()
        logger.success(f"Lemmatizer successful. Time taken: {end-start}")
        return query_output
