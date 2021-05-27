import unittest

from dynaconf import settings

from bayes_opt_qa.tasks.data_extraction import FactClassificationTask
from poly_nlp.tasks.datasets.genericskb import GenericsKBExtractionTask


class FactClassificationTest(unittest.TestCase):
    def test_fact_classification(self):
        generics_kb_path = settings["generics_kb_test"]
        dataset = GenericsKBExtractionTask().run(generics_kb_path)
        classified = FactClassificationTask().run(dataset)
