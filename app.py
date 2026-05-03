from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model("bodyfat_model.keras")
with open("scaler.pkl", "rb") as f:
    sc = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age     = float(request.form["age"])
    weight  = float(request.form["weight"])
    height  = float(request.form["height"])
    abdomen = float(request.form["abdomen"])
    chest   = float(request.form["chest"])
    hip     = float(request.form["hip"])

    # Convert kg to lbs for model
    weight_lbs = weight * 2.205

    features = [
        1.055,
        age,
        weight_lbs,
        height,
        36.0,
        chest,
        abdomen,
        hip,
        55.0,
        38.0,
        22.0,
        32.0,
        28.0,
        17.0
    ]

    data = np.array([features])
    data_scaled = sc.transform(data)
    prediction = model.predict(data_scaled)
    result = round(float(prediction[0][0]), 2)

    # Send values back to HTML so fields stay filled
    return render_template("index.html",
                           prediction=result,
                           age=age,
                           weight=weight,
                           height=height,
                           abdomen=abdomen,
                           chest=chest,
                           hip=hip)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")