import shutil
from pathlib import Path
from typing import Dict

from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm
from whoosh.analysis import RegexTokenizer, StemmingAnalyzer, StopFilter
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in


class WhooshBuildIndexTask(Task):
    @overrides
    def run(self, corpus: Dict[str, str], index_dir: str):
        path = Path(index_dir)
        stem_ana = StemmingAnalyzer() | StopFilter()
        if not path.exists():
            logger.info(f"Checkpoint dir {path} does not exist. Creating one")
            path.mkdir(parents=True)
        else:
            logger.info(f"Checkpoint dir {path} exist. Cleaning one")
            shutil.rmtree(path)
            path.mkdir(parents=True)
        schema = Schema(id=TEXT(stored=True), text=TEXT(analyzer=stem_ana, stored=True))
        ix = create_in(index_dir, schema)
        writer = ix.writer()

        for id, text in tqdm(corpus.items(), "Building indexes"):
            writer.add_document(id=id, text=text)
        writer.commit()
        return ix
