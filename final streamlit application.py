import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== Step 0: Download NLTK Stopwords ==========
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ========== Step 1: Preprocessing Functions ==========

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(df):
    df = df.dropna(subset=['Storyline'])
    df['Cleaned_Storyline'] = df['Storyline'].apply(clean_text)
    df = df.drop_duplicates(subset='Cleaned_Storyline', keep='first')  # Drop similar plot duplicates
    df = df.drop_duplicates(subset='Movie Name', keep='first')         # Drop movie name duplicates
    return df

def compute_similarity(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['Cleaned_Storyline'])
    return tfidf, tfidf_matrix

# ========== Step 2: Recommendation Function ==========

def recommend_movies_by_storyline(user_storyline, df, tfidf_vectorizer, tfidf_matrix):
    cleaned_input = clean_text(user_storyline)
    user_vector = tfidf_vectorizer.transform([cleaned_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Sort indices by similarity score in descending order
    top_indices = similarity_scores.argsort()[::-1]

    # Get unique movie recommendations
    seen = set()
    unique_recommendations = []
    for idx in top_indices:
        movie = df.iloc[idx]['Movie Name']
        if movie not in seen:
            seen.add(movie)
            unique_recommendations.append(idx)
        if len(unique_recommendations) == 5:
            break

    return df.iloc[unique_recommendations][['Movie Name', 'Storyline']]

# ========== Step 3: Streamlit App ==========

def run_streamlit(df, tfidf_vectorizer, tfidf_matrix):
    st.set_page_config(page_title="IMDb Storyline Recommender", layout="centered")

    try:
        st.image("C:/Users/Mogeeth.M/Downloads/imdb image.png", use_container_width=True)
    except:
        pass

    st.title("🎬 IMDb 2024 Movie Recommender")
    st.write("Enter a movie **storyline**, and get the top 5 most similar movies based on plot.")

    user_input_storyline = st.text_area("Enter a movie storyline:")

    if st.button("Recommend"):
        if not user_input_storyline.strip():
            st.warning("Please enter a storyline.")
        else:
            with st.spinner("Finding similar movies..."):
                results = recommend_movies_by_storyline(user_input_storyline, df, tfidf_vectorizer, tfidf_matrix)

                st.success("Top 5 Recommended Movies Based on Your Storyline:")
                for i, (idx, row) in enumerate(results.iterrows()):
                    st.markdown(f"### {idx}. {row['Movie Name']}")
                    st.write(row['Storyline'])

# ========== Step 4: Main Execution Block ==========

if __name__ == "__main__":
    try:
        df = pd.read_csv(r"C:\Users\Mogeeth.M\Downloads\IMDB\env\imdb_movies.csv")
    except FileNotFoundError:
        st.error("Error: 'imdb_movies.csv' not found. Please ensure the file path is correct.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the CSV file: {e}")
        st.stop()

    df = preprocess_data(df)
    tfidf_vectorizer, tfidf_matrix = compute_similarity(df)
    run_streamlit(df, tfidf_vectorizer, tfidf_matrix)
