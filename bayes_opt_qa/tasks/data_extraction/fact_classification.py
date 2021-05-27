import spacy
from nltk.corpus import stopwords
from overrides import overrides
from prefect import Task
from tqdm import tqdm

from poly_nlp.parallel.ray_executor import RayExecutor

GROUNDING = "GROUNDING"
ABSTRACT = "ABSTRACT"

PATTERNS = ["is", "are", "means", "mean"]


class FactClassificationTask(Task):
    @staticmethod
    def process_facts(pos, input):
        classified_facts = {}
        stop_words = set(stopwords.words("english"))

        nlp = spacy.load("en_core_web_sm")
        for id, fact_info in input.items():
            fact = fact_info["fact"].lower()

            doc = nlp(fact.lower())
            verb_tokens = [token.text for token in doc if "VB" in token.tag_]
            # if len(arg1) <= count and len(arg2) <= count:
            if len(verb_tokens) == 1 and verb_tokens[0] in PATTERNS:
                arg1 = [
                    word
                    for word in fact.split(verb_tokens[0])[0].split()
                    if word not in stop_words
                ]
                arg2 = [
                    word
                    for word in fact.split(verb_tokens[0])[1].split()
                    if word not in stop_words
                ]
                # print(fact, "GROUNDING")
                classified_facts[id] = {
                    "type": GROUNDING,
                    "arg1": " ".join(arg1),
                    "arg2": " ".join(arg2),
                }
            else:
                # print(fact, "ABSTRACT")
                classified_facts[id] = {"type": ABSTRACT}
        return classified_facts

    @overrides
    def run(self, dataset):
        ray_executor = RayExecutor()
        return ray_executor.run(dataset, self.process_facts, {},)
