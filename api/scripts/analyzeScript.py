from unittest import result
from newspaper import Article

def analyzeArticle(url):
    print("in script url: ", url)
    article = Article(url)
    article.download()
    article.parse()
    headline = article.title
    img = article.top_image
    text = article.text
    result = {
        "headline": headline,
        "img": img,
        "text": text
    }
    print(result)
    return result