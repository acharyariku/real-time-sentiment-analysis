import feedparser
from kafka_client.producer import get_kafka_producer 

def fetch_and_send_rss():
    rss_url = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
    feed = feedparser.parse(rss_url)

    producer = get_kafka_producer()

    for entry in feed.entries:
        message = {
            "headline": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", None),
        }
        producer.send('news_headlines',value=message)

    producer.flush()

if __name__ == "__main__":
    fetch_and_send_rss()