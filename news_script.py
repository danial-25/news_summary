import newsapi
from newsapi.newsapi_client import NewsApiClient
newsapi=NewsApiClient(api_key='a2b489b27aee49b6a39c44b9fd95428c')

def get_news(categories=None):
    news=[]
    if categories is None:
        newsapi=NewsApiClient(api_key='a2b489b27aee49b6a39c44b9fd95428c')

        top_headlines = newsapi.get_top_headlines(
                                                sources='bbc-news,the-verge',
                                                page=3,
                                                language='en',
                                                )
        return top_headlines['articles'][:3]
    else:
        newsapi=NewsApiClient(api_key='a2b489b27aee49b6a39c44b9fd95428c')

        for category in categories:
            if category is not None:
                top_headlines = newsapi.get_top_headlines(
                                                        page=3,
                                                        category=category,
                                                        language='en',
                                                        )
                news.append(top_headlines['articles'][:3])
            else: 
                newsapi=NewsApiClient(api_key='a2b489b27aee49b6a39c44b9fd95428c')

                top_headlines = newsapi.get_top_headlines(
                                                sources='bbc-news,the-verge',
                                                page=3,
                                                language='en',
                                                )
                news.append(top_headlines['articles'][:3])
        return news