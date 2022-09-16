import flask as f
from pandas import DataFrame
from pickle import load
import os

# # Create a web application and load contents of pickle file
app = f.Flask(__name__)
model = load(open("model.pkl", "rb"))
scater = load(open("scater.pkl", "rb"))


# Load template on opening app home page
@app.route("/")
def home():
    return f.render_template("index.html")


# Create predictions and show it on page
@app.route("/predict", methods=["POST"])
def predict():
    A = []
    for i in f.request.form.values():
        A.append(float(i))
    predicted_score = round(model.predict([[A[0]]])[0][0], 1)
    return f.render_template("result.html", pred=predicted_score, A=A[0])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
