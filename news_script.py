import newsapi
from newsapi.newsapi_client import NewsApiClient
from dotenv import load_dotenv
import os

load_dotenv()
NEWS_API = os.getenv("NEWS_API")
newsapi = NewsApiClient(api_key=NEWS_API)


def get_news(categories=None):
    news = []
    if categories is None:
        newsapi = NewsApiClient(api_key=NEWS_API)

        top_headlines = newsapi.get_top_headlines(
            sources="bbc-news,the-verge",
            page=3,
            language="en",
        )
        return top_headlines["articles"][:3]
    else:
        newsapi = NewsApiClient(api_key=NEWS_API)

        for category in categories:
            if category is not None:
                top_headlines = newsapi.get_top_headlines(
                    page=3,
                    category=category,
                    language="en",
                )
                news.append(top_headlines["articles"][:3])
            else:
                newsapi = NewsApiClient(api_key=NEWS_API)

                top_headlines = newsapi.get_top_headlines(
                    sources="bbc-news,the-verge",
                    page=3,
                    language="en",
                )
                news.append(top_headlines["articles"][:3])
        return news
