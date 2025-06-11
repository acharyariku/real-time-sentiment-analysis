from kafka import KafkaConsumer
import json

def get_kafka_consumer():
    consumer = KafkaConsumer(
        'news_headlines',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        group_id='my-consumer-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        key_deserializer=lambda x: json.loads(x.decode('utf-8')),
    )
    return consumer

def consume_messages():
    consumer = get_kafka_consumer()
    print("Consumer started. Waiting for messages...")
    for msg in consumer:
        print(f"Received message: key={msg.key}, value={msg.value}")

if __name__ == "__main__":
    consume_messages()