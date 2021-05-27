import unittest

from dynaconf import settings

from bayes_opt_qa.tasks.explanation_construction import EntityExtractionTask
from bayes_opt_qa.tasks.utils import WorldTreeLemmatizer


class EntityExtractionTaskTest(unittest.TestCase):
    def test_entity_extraction(self):
        entities = EntityExtractionTask().run(
            {
                "id1": "friction occurs	when	two	object's	surfaces move	against	each	other",
                "id2": "the full moon occurs every 13 days",
                "id3": "friction is a kind of force",
            }
        )
        print(entities)

    @unittest.skip
    def test_lemmatizer_extraction(self):
        lemmatizer_path = settings[f"worldtree_v1"]["lemmatizer_path"]
        entities = WorldTreeLemmatizer().run(
            {
                "id1": "friction occurs	when	two	object's	surfaces move	against	each	other",
                "id2": "the full moon occurs every 13 days",
            },
            lemmatizer_path,
        )
        print(entities)
