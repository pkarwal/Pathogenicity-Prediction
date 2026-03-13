from flask import Flask, render_template, request
import pandas as pd
import joblib
from utils import parse_cds_mutation, parse_aa_mutation

app = Flask(__name__)

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    error = None

    if request.method == "POST":

        cds_mut = request.form.get("cds_mut")
        aa_mut = request.form.get("aa_mut")

        if not cds_mut or not aa_mut:
            error = "Please enter both mutations."
        else:
            try:

                cds_pos, cds_from, cds_to = parse_cds_mutation(cds_mut)
                aa_pos, aa_from, aa_to = parse_aa_mutation(aa_mut)

                input_dict = {
                    "cds_pos":[cds_pos],
                    "cds_from":[cds_from],
                    "cds_to":[cds_to],
                    "aa_pos":[aa_pos],
                    "aa_from":[aa_from],
                    "aa_to":[aa_to]
                }

                df = pd.DataFrame(input_dict)

                df = pd.get_dummies(
                    df,
                    columns=["cds_from","cds_to","aa_from","aa_to"]
                )

                df = df.reindex(columns=feature_columns, fill_value=0)

                pred = model.predict(df)[0]
                prob = model.predict_proba(df)[0][1]

                probability = round(prob * 100, 2)
                prediction = "Pathogenic" if pred == 1 else "Benign"

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == "__main__":
    app.run()