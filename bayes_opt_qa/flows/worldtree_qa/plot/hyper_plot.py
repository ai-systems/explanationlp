import pandas as pd
import plotly.express as px
import ujson as json

paths = [
    "data/checkpoint/worldtree_worldtree_v1_answer/answer_selection_uni_dev.json",
    "data/checkpoint/worldtree_worldtree_v1_answer/answer_selection_uni_dev_2.json",
]

sample = {
    "abstraction_overlap": -1.0,
    "abstraction_question_overlap": 1.0,
    "abstraction_relevance": 0.17716039062054365,
    "unification_abstraction_overlap": 0.0,
    "challenge_explanations": 2.3,
    "unification_overlap": -0.6232498078299699,
    "unification_question_overlap": 1.0,
    "unification_relevance": 1.0,
    "wrong_overlap": 0.0,
}
for path in paths:
    final_dict = {
        "G-G(O)": [],
        "G-Q(O)": [],
        "G(R)": [],
        "G-C(O)": [],
        "C-C(O)": [],
        "C-Q(O)": [],
        "C(R)": [],
        "Choice Overlap": [],
        "Fact Count": [],
        "Accuracy (%)": [],
    }
    dimensions = list(
        [
            dict(range=[-1, 0], label="G-G(O)"),
            dict(range=[0, 1], label="G-Q(O)"),
            dict(range=[0, 1], label="G(R)"),
            dict(range=[0, 1], label="G-C(O)"),
            dict(range=[0, 1], label="C-C(O)"),
            dict(range=[0, 1], label="C-Q(O)"),
            dict(range=[0, 1], label="C(R)"),
            dict(range=[-1, 0], label="Choice Overlap"),
            dict(range=[0.7, 0.3], label="Accuracy"),
            dict(range=[5, 2], label="Fact Count"),
        ]
    )
    with open(path, "r") as f:
        df_data = {}
        for line in f:
            data = json.loads(line)
            params = data["params"]
            score = data["target"]
            if (score > 0.685 and score < 0.7) or (score < 0.40 and score > 0.35):
                updated_val = {
                    "G-G(O)": round(params["abstraction_overlap"], 2),
                    "G-Q(O)": round(params["abstraction_question_overlap"], 2),
                    "G(R)": round(params["abstraction_relevance"], 2),
                    "G-C(O)": float(
                        "%.2f" % round(params["unification_abstraction_overlap"], 2)
                    ),
                    "C-C(O)": round(params["unification_overlap"], 2),
                    "C-Q(O)": round(params["unification_question_overlap"], 2),
                    "C(R)": round(params["unification_relevance"], 2),
                    "Choice Overlap": round(params["wrong_overlap"], 2),
                    "Fact Count": round(params["challenge_explanations"], 2),
                    "Accuracy (%)": round(score, 2) * 100,
                }
                for key, val in updated_val.items():
                    final_dict[key].append(val)
        final_df = pd.DataFrame.from_dict(final_dict)
        cols = set(final_dict.keys())

        fig = px.parallel_coordinates(
            final_df,
            color="Accuracy (%)",
            dimensions=list(cols),
            color_continuous_scale=px.colors.diverging.Tealrose,
            color_continuous_midpoint=50,
            range_color=[70, 40],
        )
        fig.write_html("output.html")
        # fig.write_image("output.png")
