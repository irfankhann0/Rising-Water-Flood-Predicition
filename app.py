from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Model
model = joblib.load("model/model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        data = [[
            float(request.form["JAN"]),
            float(request.form["FEB"]),
            float(request.form["MAR"]),
            float(request.form["APR"]),
            float(request.form["MAY"]),
            float(request.form["JUN"]),
            float(request.form["JUL"]),
            float(request.form["AUG"]),
            float(request.form["SEP"]),
            float(request.form["OCT"]),
            float(request.form["NOV"]),
            float(request.form["DEC"]),
            float(request.form["Jan-Feb"]),
            float(request.form["Mar-May"]),
            float(request.form["Jun-Sep"]),
            float(request.form["Oct-Dec"])
        ]]

        data = np.array(data)

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0]

        confidence = round(max(probability) * 100, 2)
        if prediction == 1:
            result = "⚠️ Flood Likely"
        else:
            result = "✅ No Flood Risk"

        return render_template(
    "result.html",
    result=result,
    confidence=confidence
)

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)