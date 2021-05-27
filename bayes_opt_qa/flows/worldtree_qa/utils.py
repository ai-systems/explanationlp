from loguru import logger
from prefect import task


@task
def table_store_mapping(table_store, to_filter=None, to_filter2=None):

    if to_filter is not None:
        data = set()
        for t_ids in to_filter.values():
            data.update(list(t_ids.keys()))
        if to_filter2 is not None:
            for t_ids in to_filter2.values():
                data.update(list(t_ids.keys()))
        logger.info(f"Data length size {len(data)}")
        return {id: fact["fact"] for id, fact in table_store.items() if id in data}
    else:
        return {id: fact["fact"] for id, fact in table_store.items()}


@task
def table_score_mapping_fill(table_store):
    updated = {}
    for id, fact in table_store.items():
        fill_fact = ""
        for header, content in fact["explanation"].items():
            if "FILL" not in header:
                fill_fact = f"{fill_fact} {content}"
        updated[id] = fill_fact
    return updated


@task
def worldtree_mapping(worldtree):
    return {
        id: f"{question_exp['question']} {question_exp['answer']}"
        for id, question_exp in worldtree.items()
    }


@task
def explanation_mapping(worldtree, table_store):

    table_mapping = {}
    for id, fact in table_store.items():
        table_mapping[fact["id"]] = fact["table_name"]

    return {
        id: [
            e_id
            for e_id in question_exp["explanation"].keys()
            if table_mapping[e_id] not in ["KINDOF", "SYNONYMY"]
        ]
        for id, question_exp in worldtree.items()
    }


@task
def result_mapping(result, table_store):
    output = {
        id: list(
            [
                table_store[r]["id"]
                for r in res.keys()
                if table_store[r]["table_name"] not in ["KINDOF", "SYNONYMY"]
            ]
        )[:5]
        for id, res in result.items()
    }
    return output


@task
def result_mapping_with_score(result):
    output = {id: [(r["id"], r["score"]) for r in res] for id, res in result.items()}
    return output


@task
def analyze_entities(
    question_entities,
    table_store_entites,
    world_tree,
    table_store,
    results,
    table_results,
):
    score_mapping = defaultdict(
        lambda: 0,
        {
            q_id: {res["id"]: res["score"] for res in predicted}
            for q_id, predicted in results.items()
        },
    )
    table_score_mapping = {
        t_id: {res["id"]: res["score"] for res in predicted}
        for t_id, predicted in table_results.items()
    }
    selected_explanations = {}
    for q_id, question_exp in tqdm(world_tree.items()):
        q_entities = set(question_entities[q_id])
        fact_scores = score_mapping[q_id]

        print(f'{question_exp["question"]} {question_exp["answer"]}')
        print(q_entities)

        for t_id in fact_scores:
            t_entites = set(table_store_entites[t_id])
            q_coverage = len(q_entities.intersection(t_entites)) / len(q_entities)
            q_divergence = (
                len(t_entites - q_entities) / len(q_entities)
                if len(t_entites) > 0
                else 0
            )
            fact_scores[t_id] = fact_scores[t_id] + 0.2 * q_coverage
        sorted_facts = {
            t_id: score
            for t_id, score in sorted(fact_scores.items(), key=lambda item: item[1])
        }
        top_explanations = list(reversed(list(sorted_facts.keys())))[:30]

        # for id1 in top_explanations:
        #     for n_id, t_score in table_score_mapping[id1].items():
        #         if id1 != n_id and n_id in top_explanations:
        #             sorted_facts[n_id] += 0.01 * t_score
        # sorted_facts = {
        #     t_id: score
        #     for t_id, score in sorted(sorted_facts.items(), key=lambda item: item[1])
        # }
        # top_explanations = list(reversed(list(sorted_facts.keys())))[:30]

        selected_explanations[q_id] = top_explanations

        new_exp = defaultdict(lambda: 0)
        for t_id in top_explanations:
            q_score = sorted_facts[t_id] if t_id in sorted_facts else 0
            for n_id, t_score in table_score_mapping[t_id].items():
                if n_id not in selected_explanations[q_id]:
                    t_coverage = (
                        len(
                            set(table_store_entites[t_id]).intersection(
                                set(table_store_entites[n_id])
                            )
                        )
                        / len(set(table_store_entites[t_id]))
                        if len(set(table_store_entites[t_id])) > 0
                        else 0
                    )
                    new_exp[n_id] = new_exp[n_id] + q_score * t_score

        new_sorted_facts = {
            t_id: score
            for t_id, score in sorted(new_exp.items(), key=lambda item: item[1])
        }
        top_explanations = list(reversed(list(new_sorted_facts.keys())))
        selected_explanations[q_id] += top_explanations

    return selected_explanations


@task
def map_sample(worldtree):
    samples = [
        "Mercury_SC_401178",
        #     "ACTAAP_2013_5_16",
        #     "Mercury_SC_409390",
        #     "Mercury_SC_401244",
        #     "Mercury_SC_400612",
        # "Mercury_SC_415366",
        #     "Mercury_SC_LBS10783"
    ]
    return {id: question_exp for id, question_exp in worldtree.items() if id in samples}
