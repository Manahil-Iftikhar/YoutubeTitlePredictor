from googleapiclient.discovery import build
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
# Your API key here
API_KEY = 'AIzaSyA-3wl7lBbC3K3TDXjCm9s6Odm9In1Ei_w'
BASE_URL = "https://www.googleapis.com/youtube/v3/videos"

# Build a resource object for interacting with the YouTube API
youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_videos(query, max_results=10):

    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]
    return get_video_details(video_ids)

def get_video_details(video_ids):

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

# Example usage:
if __name__ == "__main__":
    query = input("Enter search query: ")
    video_data_df = search_videos(query, max_results=5)

    # Print video data
    print(video_data_df)

    # Optionally, save the data to a CSV file
    video_data_df.to_csv('./youtube_video_data.csv', index=False)
# Categories (example: 10 for Music, 20 for Gaming, 25 for News, 17 for Sports)
categories = {
    "Music": 10,
    "Gaming": 20,
    "News": 25,
    "Sports": 17
}

# Function to get trending videos for a specific category
def get_trending_videos(category_id, api_key):
    params = {
        'part': 'snippet,statistics',
        'chart': 'mostPopular',
        'regionCode': 'US',
        'videoCategoryId': category_id,
        'maxResults': 10,  # Number of videos to fetch
        'key': api_key
    }
    response = requests.get(BASE_URL, params=params)
    return response.json()

# Initialize an empty list to store video data
video_data = []

# Loop through categories and fetch video data
for category, category_id in categories.items():
    videos = get_trending_videos(category_id, API_KEY)
    for video in videos['items']:
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

# Convert the list to a DataFrame
df = pd.DataFrame(video_data)

# Save to CSV
df.to_csv('./trending_videos_metadata.csv', index=False)

print("Data collected and saved to trending_videos_metadata.csv")

# Load the dataset from the CSV file
df = pd.read_csv('./trending_videos_metadata.csv')

# 1. Remove duplicates
df.drop_duplicates(inplace=True)

# 2. Handle missing values
# Check for missing values in each column
print("Missing values before cleaning:")
print(df.isnull().sum())

# a. Handle missing values in the 'title' column (Drop rows with missing titles)
df = df[df['title'].notna()]

# b. Handle missing values in numerical columns (views, likes, comments)
# You can either drop these rows or fill them with zeros or mean/median values
df['views'] = df['views'].fillna(0)
df['likes'] = df['likes'].fillna(0)
df['comments'] = df['comments'].fillna(0)

# 3. Convert data types
# Convert views, likes, and comments to numeric (in case they are not)
df['views'] = pd.to_numeric(df['views'], errors='coerce')
df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
df['comments'] = pd.to_numeric(df['comments'], errors='coerce')

# Check for any issues in the numeric conversion
print("Data types after conversion:")
print(df.dtypes)

# 4. Handle categorical values (if necessary)
# For example, ensure the 'category' column is consistent
# If some categories are missing, you can label them as 'Unknown'
df['category'] = df['category'].fillna('Unknown')

# 5. Remove any rows that still have missing values
# (This step is optional depending on how strict you want to be)
df.dropna(inplace=True)

# Check if any missing values remain after cleaning
print("Missing values after cleaning:")
print(df.isnull().sum())

# 6. Save the cleaned dataset
df.to_csv('youtube_video_metadata_cleaned.csv', index=False)

print("Data cleaning is complete. Cleaned data saved to 'youtube_video_metadata_cleaned.csv'.")

# Load the YouTube video metadata CSV file
# df = pd.read_csv('youtube_video_metadata_cleaned.csv')

# Extract titles, descriptions, and tags for keyword analysis
video_text = df['title'] + ' ' + df['description']

# Convert text to lowercase and remove missing values
video_text = video_text.fillna('').str.lower()

# Tokenize and clean the text (remove stop words)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize CountVectorizer with the correct stop_words parameter
vectorizer = CountVectorizer(stop_words='english')  # Use 'english' or a list of stop words

# Assuming video_text is defined and contains the text data
X = vectorizer.fit_transform(video_text)

# Get feature names (keywords)
keywords = vectorizer.get_feature_names_out()

# Count frequency of each keyword
keyword_counts = X.sum(axis=0).A1
keyword_df = pd.DataFrame({'keyword': keywords, 'count': keyword_counts})

# Sort keywords by frequency
keyword_df = keyword_df.sort_values(by='count', ascending=False)

# Display top 10 most frequent keywords
print("Top 10 Keywords:")
print(keyword_df.head(10))


# Load your cleaned YouTube data
df = pd.read_csv('./youtube_video_metadata_cleaned.csv')

# Check if 'search_volume' column exists
if 'search_volume' not in df.columns:
    print("Warning: 'search_volume' column not found. Initializing with zeros.")
    df['search_volume'] = 0  # Initialize with zeros or any default value

# Fill missing values in the 'search_volume' column
df['search_volume'] = df['search_volume'].fillna(0)  # Replace with real search volume data

# Feature Engineering: Calculate engagement score (for example: views + likes + comments)
df['engagement_score'] = df['views'] + df['likes'] + df['comments']

# Extract relevant features (e.g., search volume, competition, title length)
df['title_length'] = df['title'].apply(len)

# Example competition and search volume columns (dummy data)
df['competition'] = df['competition'].fillna(0)  # Replace with real competition data
df['search_volume'] = df['search_volume'].fillna(0)  # Replace with real search volume data

# Final features to use for prediction
features = ['title_length', 'competition', 'search_volume']
target = 'engagement_score'

X = df[features]
y = df[target]


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict the engagement score
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression Mean Squared Error: {mse}")


# Initialize and train the SVR model
svr = SVR(kernel='linear')  # Use linear kernel for simplicity
svr.fit(X_train, y_train)

# Predict the engagement score
y_pred_svr = svr.predict(X_test)

# Evaluate the SVR model
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"SVR Mean Squared Error: {mse_svr}")


# Initialize and train the neural network
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predict the engagement score
y_pred_mlp = mlp.predict(X_test)

# Evaluate the neural network model
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print(f"Neural Network Mean Squared Error: {mse_mlp}")


# Evaluate R² score for each model
r2_lr = r2_score(y_test, y_pred)
r2_svr = r2_score(y_test, y_pred_svr)
r2_mlp = r2_score(y_test, y_pred_mlp)

print(f"Linear Regression R²: {r2_lr}")
print(f"SVR R²: {r2_svr}")
print(f"Neural Network R²: {r2_mlp}")

# Example: Predict engagement for a new video title
new_title = "Top 10 Tips for Learning Python in 2024"
new_title_length = len(new_title)

# You would fetch real competition and search volume data here
new_competition = 200  # Example value
new_search_volume = 10000  # Example value

# Create a feature vector for the new data
new_data = [[new_title_length, new_competition, new_search_volume]]

# Use the trained model to predict engagement
predicted_engagement = lr.predict(new_data)
print(f"Predicted Engagement Score for '{new_title}': {predicted_engagement[0]}")
