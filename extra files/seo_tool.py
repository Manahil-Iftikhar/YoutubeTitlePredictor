from googleapiclient.discovery import build
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import asyncio
import aiohttp

# API configuration
API_KEY = 'AIzaSyA-3wl7lBbC3K3TDXjCm9s6Odm9In1Ei_w'
BASE_URL = "https://www.googleapis.com/youtube/v3/videos"

youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_videos(query, max_results=10):
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    ).execute()
    video_ids = [item['id']['videoId'] for item in search_response['items']]
    return video_ids 

def get_video_details(video_ids):
    try:
        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=','.join(video_ids)
        ).execute()
        
        video_data = []
        for item in video_response['items']:
            video_info = {
                'Title': item['snippet']['title'],
                'Description': item['snippet']['description'],
                'Tags': item['snippet'].get('tags', []),
                'View Count': item['statistics'].get('viewCount', 0),
                'Like Count': item['statistics'].get('likeCount', 0),
                'Comment Count': item['statistics'].get('commentCount', 0),
                'Video URL': f"https://www.youtube.com/watch?v={item['id']}"
            }
            video_data.append(video_info)
        
        return pd.DataFrame(video_data)
    except Exception as e:
        logging.error(f"Error fetching video details: {str(e)}")
        return pd.DataFrame()

async def get_trending_videos_async(session, category_id, api_key):
    params = {
        'part': 'snippet,statistics',
        'chart': 'mostPopular',
        'regionCode': 'US',
        'videoCategoryId': category_id,
        'maxResults': 10,
        'key': api_key
    }
    async with session.get(BASE_URL, params=params) as response:
        return await response.json()

async def fetch_trending_videos_async(categories, api_key):
    async with aiohttp.ClientSession() as session:
        tasks = [get_trending_videos_async(session, category_id, api_key) for category, category_id in categories.items()]
        results = await asyncio.gather(*tasks)
    
    video_data = []
    for category, result in zip(categories.keys(), results):
        for video in result['items']:
            video_info = {
                'category': category,
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'views': video['statistics'].get('viewCount', 0),
                'likes': video['statistics'].get('likeCount', 0),
                'comments': video['statistics'].get('commentCount', 0),
                'publishedAt': video['snippet']['publishedAt'],
            }
            video_data.append(video_info)
    return pd.DataFrame(video_data)

def fetch_trending_videos(categories, api_key):
    return asyncio.run(fetch_trending_videos_async(categories, api_key))

def clean_data(df):
    df = df.drop_duplicates()
    df = df[df['title'].notna()]
    df[['views', 'likes', 'comments']] = df[['views', 'likes', 'comments']].fillna(0)
    df[['views', 'likes', 'comments']] = df[['views', 'likes', 'comments']].apply(pd.to_numeric, errors='coerce')
    df['category'] = df['category'].fillna('Unknown')
    return df.dropna()

def analyze_keywords(df):
    video_text = df['title'] + ' ' + df['description'] + ' ' + df['tags'].fillna('')
    video_text = video_text.str.lower()
    
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2))
    X = vectorizer.fit_transform(video_text)
    
    keywords = vectorizer.get_feature_names_out()
    keyword_counts = X.sum(axis=0).A1
    keyword_df = pd.DataFrame({'keyword': keywords, 'count': keyword_counts})
    return keyword_df.sort_values(by='count', ascending=False)

def prepare_features(df):
    df['engagement_score'] = df['views'] + df['likes'] + df['comments']
    df['title_length'] = df['title'].apply(len)
    df['competition'] = df['competition'].fillna(0)
    df['search_volume'] = df['search_volume'].fillna(0)
    
    features = ['title_length', 'competition', 'search_volume']
    target = 'engagement_score'
    
    return df[features], df[target]

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='linear'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'mse': mse, 'r2': r2}
    
    return results

def predict_engagement(model, title, competition, search_volume):
    new_data = [[len(title), competition, search_volume]]
    return model.predict(new_data)[0]

def fetch_and_save_video_data(query):
    video_ids = search_videos(query, max_results=5)
    video_data_df = get_video_details(video_ids)
    video_data_df.to_csv('youtube_video_data.csv', index=False)

def fetch_and_save_trending_videos(categories):
    trending_df = fetch_trending_videos(categories, API_KEY)
    trending_df.to_csv('trending_videos_metadata.csv', index=False)

def process_and_analyze_data():
    df = pd.read_csv('youtube_video_metadata.csv')
    df_cleaned = clean_data(df)
    df_cleaned.to_csv('youtube_video_metadata_cleaned.csv', index=False)
    
    keyword_df = analyze_keywords(df_cleaned)
    print("Top 10 Keywords:")
    print(keyword_df.head(10))
    
    return df_cleaned

def train_models_and_predict(df_cleaned):
    X, y = prepare_features(df_cleaned)
    model_results = train_and_evaluate_models(X, y)
    
    for name, result in model_results.items():
        print(f"{name} - MSE: {result['mse']}, R²: {result['r2']}")
    
    new_title = "Top 10 Tips for Learning Python in 2024"
    predicted_engagement = predict_engagement(
        model_results['Linear Regression']['model'],
        new_title,
        competition=200,
        search_volume=10000
    )
    print(f"Predicted Engagement Score for '{new_title}': {predicted_engagement}")

def main():     
    query = input("Enter search query: ")
    fetch_and_save_video_data(query)
    
    categories = {
        "Music": 10,
        "Gaming": 20,
        "News": 25,
        "Sports": 17
    }
    fetch_and_save_trending_videos(categories)
    
    df_cleaned = process_and_analyze_data()
    train_models_and_predict(df_cleaned)

if __name__ == "__main__":
    main()
