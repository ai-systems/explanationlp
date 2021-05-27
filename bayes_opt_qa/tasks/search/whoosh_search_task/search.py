from typing import Dict

import numpy as np
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm
from whoosh.index import FileIndex
from whoosh.qparser import OrGroup, QueryParser


class WhooshSearchTask(Task):
    @overrides
    def run(self, ix: FileIndex, query: Dict[str, str], normalize=False):
        search_results = {}
        logger.info(f"Searching Whoosh Index.  Normalize: {normalize}")
        for id, text in tqdm(query.items(), "Searching query"):
            with ix.searcher() as searcher:
                query_parser = QueryParser("text", ix.schema, group=OrGroup).parse(text)
                results = searcher.search(query_parser)
                result_scores = [result.score for result in results]
                if normalize:
                    result_scores = np.array(result_scores) / np.sum(result_scores)
                search_results[id] = [
                    (result["id"], score)
                    for result, score in zip(results, result_scores)
                ]
        return search_results
