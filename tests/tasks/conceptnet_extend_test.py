import unittest

from bayes_opt_qa.tasks.utils.conceptnet_extend import ConceptNetExtend
from dynaconf import settings
from poly_nlp.tasks.datasets.genericskb import GenericsKBExtractionTask


class ConceptNetExtendTest(unittest.TestCase):
    def test_conceptnet_extend(self):
        generics_kb_path = settings["generics_kb"]["best"]
        generic_kb = GenericsKBExtractionTask().run(
            "tests/tasks/GenericsKB-Best.tsv",
            filter_fn=lambda fact: fact["source"] not in ["ConceptNet"],
        )
        extended = ConceptNetExtend().run(generic_kb)
        print(extended)
