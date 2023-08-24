from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the movies data
movies_data = pd.read_csv('movies .csv')

# Perform data preprocessing and cosine similarity calculations
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']

    # Find a close match for the given movie name
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]

    # Get the index of the movie
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    # Get similarity scores for the movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort the movies based on similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Prepare the list of recommended movies (limit to 10)
    recommendations = []

    for movie in sorted_similar_movies[:10]:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommendations.append(title_from_index)

    return render_template('index.html', movie_name=movie_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
