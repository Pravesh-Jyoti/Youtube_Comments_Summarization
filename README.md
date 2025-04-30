# Youtube_Comments_Summarization
We built a YouTube comment sentiment analysis tool using Streamlit ,NLP and Machine Learning. 
Here's a quick overview:
First, we clean the text data from YouTube comments by removing unwanted characters, tokenizing, and lemmatizing the words. We also remove common stopwords to focus on meaningful words.
We trained a support vector machine to predict the sentiment of YouTube comments as Positive or Negative. The model is trained on labeled data using a TF-IDF vectorizer to transform the text into numeric features.
The model fetches comments from a YouTube video using the YouTube API, by simply entering the video URL. It retrieves up to 100 comments for analysis.
Users can input a single comment, and the model predicts whether it's positive or negative.
Users can input a video URL, and the model will analyze the sentiments of all comments from the video. It also generates a bar chart showing the distribution of sentiments and displays the top 5 positive comments.
The sentiment distribution is visualized in a bar chart, showing counts of positive and negative comments.
This project combines Natural Language Processing (NLP), Machine Learning, and Streamlit to create an interactive tool for analyzing YouTube comments.
