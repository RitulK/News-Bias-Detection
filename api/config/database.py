from pymongo import MongoClient


client = MongoClient('mongodb+srv://anishketkar05:anishUSER@news-analysis-c1.vonz85k.mongodb.net/?appName=news-analysis-c1')

db1 = client.Events
mainEvents = db1['MainEvents']

db2 = client.Articles
mainArticles = db2['MainArticles']
