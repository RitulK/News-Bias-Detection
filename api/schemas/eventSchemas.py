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
        "centroid_embedding" : event["centroid_embedding"]
    }

def indEventHeadline(event)->dict:
    return {
        "_id" : str(event["_id"]),
        "eventHeadline": event["eventHeadline"],
    }

def listEventHeadlines(events)-> list:
    return [indEventHeadline(event) for event in events]