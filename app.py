import gradio as gr
import numpy as np
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64,  activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64,  activation='relu'))
model.add(tf.keras.layers.Dense(32,  activation='relu'))
model.add(tf.keras.layers.Dense(1))

model(np.zeros((1, 14)))
model.load_weights("bodyfat_model.weights.h5")

with open("scaler.pkl", "rb") as f:
    sc = pickle.load(f)

def predict_bodyfat(age, weight, height, neck, chest,
                    abdomen, hip, thigh, knee,
                    ankle, biceps, forearm, wrist):

    # Smart density based on age
    if age < 30:
        density = 1.065
    elif age < 40:
        density = 1.060
    elif age < 50:
        density = 1.055
    elif age < 60:
        density = 1.045
    else:
        density = 1.038

    # Convert kg to lbs
    weight_lbs = weight * 2.205

    features = [
        density, age, weight_lbs, height,
        neck, chest, abdomen, hip, thigh,
        knee, ankle, biceps, forearm, wrist
    ]

    data = np.array([features])
    data_scaled = sc.transform(data)
    prediction = model.predict(data_scaled)
    result = round(float(prediction[0][0]), 2)

    # Show result with category
    if result < 8:
        category = "💪 Very Athletic!"
    elif result <= 15:
        category = "✅ Fit and Healthy!"
    elif result <= 20:
        category = "✅ Normal"
    elif result <= 25:
        category = "⚠️ Above Average"
    elif result <= 30:
        category = "⚠️ High"
    else:
        category = "❌ Very High — Please consult a doctor"

    return f"{result}% — {category}"

# Create UI
demo = gr.Interface(
    fn=predict_bodyfat,
    inputs=[
        gr.Number(label="Age (years)",      value=25),
        gr.Number(label="Weight (kg)",      value=70),
        gr.Number(label="Height (inches)",  value=70),
        gr.Number(label="Neck (cm)",        value=36),
        gr.Number(label="Chest (cm)",       value=93),
        gr.Number(label="Abdomen (cm)",     value=85),
        gr.Number(label="Hip (cm)",         value=94),
        gr.Number(label="Thigh (cm)",       value=59),
        gr.Number(label="Knee (cm)",        value=37),
        gr.Number(label="Ankle (cm)",       value=22),
        gr.Number(label="Biceps (cm)",      value=32),
        gr.Number(label="Forearm (cm)",     value=27),
        gr.Number(label="Wrist (cm)",       value=17),
    ],
    outputs=gr.Textbox(label="Predicted Body Fat %"),
    title="💪 Body Fat % Predictor",
    description="Enter your body measurements to predict body fat percentage!"
)

demo.launch()