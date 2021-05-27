import random

import dash
import dash_cytoscape as cyto
import dash_html_components as html
import msgpack

path = "data/graphs/MCAS_2003_5_31|a bear hibernating in winter"

with open(path, "rb") as f:
    data = msgpack.unpackb(f.read(), raw=False)

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        cyto.Cytoscape(
            id="cytoscape",
            elements=[
                {
                    "data": {"id": id, "label": fact},
                    "position": {
                        "x": random.randint(1, 100),
                        "y": random.randint(1, 100),
                    },
                }
                for id, fact in data["nodes"].items()
            ]
            + [
                {
                    "data": {
                        "source": ids.split("|")[0],
                        "target": ids.split("|")[1],
                        "label": score,
                    }
                }
                for ids, score in data["edges"].items()
                if score > 0
            ],
            layout={"name": "preset"},
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
