import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tmdbv3api import TMDb, Movie
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize TMDB with error handling
@st.cache_resource
def initialize_tmdb():
    tmdb = TMDb()
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        # Do not call Streamlit functions here (module import time). Raise and let caller handle UI.
        raise RuntimeError("TMDB_API_KEY not found in environment")

    tmdb.api_key = api_key
    return tmdb, Movie()

# Module-level placeholders; actual initialization happens inside main() after page config
tmdb = None
tmdb_movie = None

# Load and preprocess data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    return preprocess_data(movies, credits)

def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

def convertcast(obj):
    l = []
    c = 0
    for i in ast.literal_eval(obj)[:3]:
        if c <= 3:
            l.append(i['name'])
            c = c + 1
    return l

def convert1(obj):
    x = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            x.append(i['name'])
            break
    return x

def preprocess_data(movies, credits):
    # Merge datasets
    movies = movies.merge(credits, on='title')
    movies = movies[['genres', 'id', 'keywords', 'title', 'overview', 'cast', 'crew']]
    
    # Drop missing values and duplicates
    movies = movies.dropna()
    
    # Convert string representations to lists
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convertcast)
    movies['crew'] = movies['crew'].apply(convert1)
    
    # Process text data
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Create tags
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['overview']
    
    # Create new dataframe with required columns
    new_df = movies[['id', 'title', 'tags']]
    
    # Process tags
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    return new_df

def create_similarity_matrix(new_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    return cosine_similarity(vectors)

def get_recommendations(movie_title, new_df, similarity, n_recommendations=5):
    # Find the index of the input movie
    movie_idx = new_df[new_df['title'] == movie_title].index[0]
    
    # Get similarity scores for this movie with all other movies
    movie_similarities = similarity[movie_idx]
    
    # Get indices of movies sorted by similarity (excluding the input movie itself)
    similar_movie_indices = np.argsort(movie_similarities)[::-1][1:n_recommendations+1]
    
    recommendations = []
    for idx in similar_movie_indices:
        recommendations.append((new_df.iloc[idx]['title'], movie_similarities[idx]))
    
    return recommendations

def get_movie_details(movie_title):
    """Get movie details from TMDB API"""
    if tmdb_movie is None:
        st.error("TMDB API is not initialized properly")
        return None
        
    try:
        search = tmdb_movie.search(movie_title)
        if not search:
            st.warning(f"No results found for movie: {movie_title}")
            return None
        
        movie_id = search[0].id
        details = tmdb_movie.details(movie_id)
        
        return {
            'title': details.title,
            'poster_path': f"https://image.tmdb.org/t/p/w500{details.poster_path}" if details.poster_path else None,
            'overview': details.overview,
            'release_date': getattr(details, 'release_date', 'N/A'),
            'rating': getattr(details, 'vote_average', 'N/A'),
            'genres': [genre['name'] for genre in getattr(details, 'genres', [])],
            'runtime': getattr(details, 'runtime', 'N/A'),
        }
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Please check your internet connection.")
        return None
    except Exception as e:
        st.error(f"Error fetching movie details: {str(e)}")
        return None

def display_movie_card(movie_details, similarity_score=None, image_width=200, reverse=False):
    """Display a movie card with poster and details.

    Parameters:
    - movie_details: dict
    - similarity_score: float or None
    - image_width: int width of poster image
    - reverse: if True, place details first then image (useful for narrow layouts)
    """
    if not movie_details:
        st.warning(f"Could not fetch details for this movie")
        return

    # Choose column proportions based on image width
    img_col_ratio = 1
    details_col_ratio = max(2, int(400 / image_width))
    if reverse:
        col_left, col_right = st.columns([details_col_ratio, img_col_ratio])
    else:
        col_left, col_right = st.columns([img_col_ratio, details_col_ratio])

    # Image column
    with col_left if not reverse else col_right:
        if movie_details['poster_path']:
            st.image(movie_details['poster_path'], width=image_width)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Poster", width=image_width)

    # Details column
    with col_right if not reverse else col_left:
        st.subheader(movie_details['title'])
        if similarity_score is not None:
            st.metric("Similarity Score", f"{similarity_score:.2%}")

        st.write("**Release Date:**", movie_details['release_date'])
        st.write("**Rating:**", f"⭐ {movie_details['rating']}/10")
        st.write("**Runtime:**", f"{movie_details['runtime']} minutes")
        st.write("**Genres:**", ", ".join(movie_details['genres']))

        with st.expander("Overview"):
            st.write(movie_details['overview'])

def main():
    st.set_page_config(
        page_title="Movie Recommender System",
        page_icon="🎬",
        layout="wide"
    )
    # Initialize TMDB now that Streamlit page config is set
    global tmdb, tmdb_movie
    try:
        tmdb, tmdb_movie = initialize_tmdb()
    except Exception as e:
        st.error(f"TMDB initialization error: {e}")
        tmdb_movie = None

    st.title('🎬 Movie Recommendation System')
    st.write('This system recommends movies similar to your input using Item-Based Collaborative Filtering')
    
    # Add custom CSS
    st.markdown("""
        <style>
        .movie-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading movie data...'):
        new_df = load_data()
        similarity = create_similarity_matrix(new_df)
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Get the list of all movies
        movie_list = new_df['title'].tolist()
        
        # Create a dropdown to select a movie
        selected_movie = st.selectbox(
            'Select a movie you like:',
            movie_list
        )
        
        if selected_movie:
            selected_movie_details = get_movie_details(selected_movie)
            if selected_movie_details:
                st.write("**Selected Movie:**")
                # Use a slightly smaller poster and reverse layout for better alignment
                display_movie_card(selected_movie_details, image_width=160, reverse=True)
        
        if st.button('Get Recommendations', key='recommend_button'):
            with col2:
                st.subheader('Recommended Movies:')
                
                # Get recommendations
                recommendations = get_recommendations(selected_movie, new_df, similarity)
                
                # Display recommendations with movie details
                for movie, score in recommendations:
                    with st.container():
                        st.markdown("---")
                        movie_details = get_movie_details(movie)
                        if movie_details:
                            display_movie_card(movie_details, score)

if __name__ == '__main__':
    main()
