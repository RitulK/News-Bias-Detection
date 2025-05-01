from datetime import datetime

from bson import ObjectId

def indEvent(event)->dict:
    return {
        "eventHeadline": event["eventHeadline"],
        "location": event["location"],
        "leftSummary": event["leftSummary"],
        "centerSummary": event["centerSummary"],
        "rightSummary": event["rightSummary"],
        "lCount": event["lCount"],
        "cCount": event["cCount"],
        "rCount": event["rCount"],
        "totalArticles": event["totalArticles"],
        "publishedDate": datetime.utcnow(),  # auto timestamp
        "articles": listArticles(event["articles"]),
        # "centroid_embedding" : event["centroid_embedding"]

    }

def indListArticle(article)->dict:
    return {
        "title": article["title"],
        "description": article["description"],
        "content": article["content"],
        "imgURL": article["imgURL"],
        "link": article["link"],
        "timestamp": article["timestamp"],
        "location": article["location"],
        "source": article["source"],
        "alignment": article["alignment"],
        "event_id": article["event_id"]
    }

def listArticles(articles)-> list:
    return [indListArticle(article) for article in articles]

def indEventHeadline(event)->dict:
    return {
        "_id" : str(event["_id"]),
        "eventHeadline": event["eventHeadline"],
    }

def listEventHeadlines(events)-> list:
    return [indEventHeadline(event) for event in events]