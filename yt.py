import streamlit as st
import os
import pandas as pd
import re
import nltk
import googleapiclient.discovery
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing function
def preProcessing(features):
    processed_features = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    for sentence in features:
        sentence = str(sentence).lower()
        sentence = re.sub(r'\W', ' ', sentence)
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        
        tokens = word_tokenize(sentence)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        processed_features.append(' '.join(tokens))
    return processed_features

# Function to fetch YouTube comments
def getComments(video_id):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyAzKpVV4kI1dRQmXp7vtttK_XhI56zIVbI"
    
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
    response = request.execute()
    
    comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
    return comments

# Load and preprocess dataset
dataset = pd.read_csv('YOUTUBE_labeled.csv').sample(n=600, random_state=42)
features = preProcessing(dataset.iloc[:, 0].values)
labels = dataset.iloc[:, 2].values

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
features = vectorizer.fit_transform(features).toarray()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Define and train SVM model
model = SVC(C=10, kernel='linear', gamma='scale')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("Accuracy is",accuracy_score(y_test,y_pred))

# Function to predict sentiment
def predict_sentiment(comment):
    processed_comment = preProcessing([comment])
    vectorized_comment = vectorizer.transform(processed_comment).toarray()
    sentiment = model.predict(vectorized_comment)[0]
    return "Positive" if sentiment == 2 else "Neutral" if sentiment == 1 else "Negative"

# Streamlit UI
st.title("YouTube Comment Sentiment Analysis")

option = st.radio("Choose an option:", ["Analyze a single comment", "Analyze a YouTube video"])

if option == "Analyze a single comment":
    user_comment = st.text_area("Enter a comment:")
    if st.button("Analyze Sentiment"):
        if user_comment:
            sentiment = predict_sentiment(user_comment)
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter a comment.")

elif option == "Analyze a YouTube video":
    video_link = st.text_input("Enter a YouTube video link:")
    if st.button("Analyze Video"):
        if video_link:
            video_id = video_link.split("v=")[-1].split("&")[0]
            comments = getComments(video_id)
            processed_comments = preProcessing(comments)
            vectorized_comments = vectorizer.transform(processed_comments).toarray()
            predictions = model.predict(vectorized_comments)
            
            sentiment_counts = {"Positive": 0, "Negative": 0}
            sentiment_map = {2: "Positive", 0: "Negative"}
            comment_sentiments = [(comment, sentiment_map[pred]) for comment, pred in zip(comments, predictions)]
            
            for _, sentiment in comment_sentiments:
                sentiment_counts[sentiment] += 1
            
            top_positive_comments = [c for c, s in comment_sentiments if s == "Positive"][:5]
            st.subheader("Top 5 Positive Comments:")
            for c in top_positive_comments:
                st.write(f"- {c}")
            
            # Sentiment distribution graph
            fig, ax = plt.subplots()
            ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['red', 'green'])
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
        else:
            st.write("Please enter a valid YouTube video link.")
