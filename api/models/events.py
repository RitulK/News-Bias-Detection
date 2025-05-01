from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from models import articles

class Event(BaseModel):
    eventHeadline: str
    location: str
    leftSummary: str
    centerSummary: str
    rightSummary: str
    lCount: int
    cCount: int
    rCount: int
    totalArticles: int
    # articles: list
    # centroid_embedding: str
    # publishedDate will be set automatically


# _id
# event_id  
# articles
# cCount #
# centerSummary #
# createdAt
# eventHeadline #
# lCount #
# leftSummary #
# location #
# rCount #
# rightSummary #
# summary
# totalArticles #
# updatedAt