from kafka import KafkaProducer
import pandas as pd
import json, time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_csv("data/processed/titanic_cleaned.csv")

for _, row in df.iterrows():
    producer.send("titanic_stream", row.to_dict())
    time.sleep(1)

producer.flush()
