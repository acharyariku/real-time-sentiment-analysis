from kafka import KafkaProducer
import json

def get_kafka_producer():
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer= lambda x:json.dumps(x).encode('utf-8'),
        key_serializer= lambda x:json.dumps(x).encode('utf-8')
    )
    return producer

def send_test_message():
    producer = get_kafka_producer()
    message = {"headline": "Hello Kafka!", "source": "test"}
    producer.send('news_headlines', value=message)
    producer.flush()  # ensures all messages are sent before exiting

if __name__ == "__main__":
    send_test_message()