import unittest

from dynaconf import settings

from bayes_opt_qa.tasks.search.whoosh_search_task import (
    WhooshBuildIndexTask,
    WhooshSearchTask,
)


class WhooshSearchTest(unittest.TestCase):
    def test_whoosh_search(self):
        index_dir = settings["index_dir"]

        build_search_index = WhooshBuildIndexTask()
        search_task = WhooshSearchTask()
        docs = {
            "1": "Which of the following happens only during the adult stage of the life cycle of a frog? ",
            "3": "What kind of animal?",
            "2": "This is a second document",
        }
        ix = build_search_index.run(docs, index_dir)
        search_query = {
            "id1": "frog is kind of an animal",
            "id2": "it only happens",
            "id4": "document",
        }
        results = search_task.run(ix, search_query)
        print(results)
