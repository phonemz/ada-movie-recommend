from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv("tmdb.csv")

# Preprocess dataset
dataset["overview"] = dataset["overview"].fillna("")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Fit and transform TF-IDF Vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(dataset["overview"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form["movie_name"]

    # Transform user input using TF-IDF Vectorizer
    movie_tfidf = tfidf_vectorizer.transform([movie_name])

    # Calculate cosine similarity between user input and dataset
    cosine_similarities = linear_kernel(movie_tfidf, tfidf_matrix).flatten()

    # Get top 5 most similar movies
    related_movies_indices = cosine_similarities.argsort()[:-11:-1]
    related_movies = dataset.iloc[related_movies_indices]["original_title"].tolist()

    return render_template(
        "recommendation.html", movie_name=movie_name, related_movies=related_movies
    )


if __name__ == "__main__":
    app.run(debug=True)
