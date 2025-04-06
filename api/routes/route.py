from fastapi import APIRouter
from models.events import Event
from schemas.eventSchemas import indEvent, listEventHeadlines, indEventHeadline
from models.articles import Article
from schemas.articleSchema import indArticle, listArticles
from config.database import mainEvents,mainArticles
from bson import ObjectId

router = APIRouter()

#GET an Event
@router.get("/event/{id}")
async def get_event(id: str):
    return indEvent(mainEvents.find_one({"_id": ObjectId(id)}))

#GET All Event Headlines
@router.get("/eventheadlines")
async def get_eventData():
    return listEventHeadlines(mainEvents.find())

#UPDATE an Event
@router.put("/event/{id}")
async def update_event(id: str, event: Event):
    mainEvents.update_one(
        {"_id": ObjectId(id)}, 
        {"$set": dict(event)}
    )
    return {"message": f"Event with ID {id} updated successfully"}

#GET All Event Headlines
@router.get("/eventembeddings")
async def get_eventEmbeddings():
    events = mainEvents.find({}, {"_id": 1, "embedding": 1})
    return [{"id": str(event["_id"]), "embedding": event["embedding"]} for event in events]

#GET an EventData
@router.get("/eventdata/{id}")
async def get_eventData(id: str):
    eventData = indEvent(mainEvents.find_one({"_id": ObjectId(id)}))
    articles = listArticles(mainArticles.find({"eventID": id}))
    return {
        "eventData" : eventData,
        "eventArticles" : articles
    }

#POST an Event
@router.post("/event")
async def post_event(event: Event):
    mainEvents.insert_one(indEvent(dict(event)))

#GET all Articles
@router.get("/article")
async def get_all_articles():
    return listArticles(mainArticles.find())

#POST an Article
@router.post("/article")
async def post_article(article: Article):
    mainArticles.insert_one(indArticle(dict(article)))