import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('datasets/tmdb_5000_movies.csv')
credits = pd.read_csv('datasets/tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords']]

# Remove missing values
movies.dropna(inplace=True)

# Create tags column
movies['tags'] = movies['overview']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')

vectors = tfidf.fit_transform(movies['tags']).toarray()

# Cosine Similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):

    movie = movie.lower()

    movies['title_lower'] = movies['title'].str.lower()

    if movie not in movies['title_lower'].values:
        return ["Movie not found"]

    index = movies[movies['title_lower'] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations


# Test
print(recommend('Avatar'))