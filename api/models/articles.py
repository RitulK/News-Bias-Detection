from bson import ObjectId
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    description: str
    content: str
    imgURL: str
    eventName: str 
    eventID: str#
    alignment: str
    source: str
    sourceID: str#
    sourceLogo: str
    link: str
    timestamp: str
    location: str
    embedding: str