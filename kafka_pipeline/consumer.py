from kafka import KafkaConsumer
import json
import pandas as pd
import pickle

from backend.preprocessing import prepare_features

model = pickle.load(open("models/survival_model.pkl", "rb"))
model_features = model.feature_names_in_

consumer = KafkaConsumer(
    "titanic_stream",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print(" Kafka consumer started. Listening for passengers...")

for message in consumer:
    data = message.value

    df = pd.DataFrame([data])

    X = prepare_features(df, model_features)

    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    print(
        f"Passenger: {data['Name']} | "
        f"Survival Probability: {prob:.2f} | "
        f"Prediction: {'Survive' if pred else 'Not Survive'}"
    )
