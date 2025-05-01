from pymongo import MongoClient


client = MongoClient('mongodb+srv://anishketkar05:anishUSER@news-analysis-c1.vonz85k.mongodb.net/?appName=news-analysis-c1')

db1 = client.news_analysis
mainEvents = db1['events']

# db2 = client.Articles
mainArticles = db1['articles']
