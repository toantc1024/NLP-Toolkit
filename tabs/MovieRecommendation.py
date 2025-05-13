import streamlit as st
import os
import pandas as pd
import requests
from qdrant_client import models, QdrantClient
from qdrant_client.http.models import PointStruct, SparseVector, NamedSparseVector
from collections import defaultdict
from dotenv import load_dotenv

def app():
    load_dotenv()

    # --- Configuration ---
    omdb_api_key = os.getenv("OMDB_API_KEY", "f4b235cc")  # Fallback key for demonstration
    collection_name = "movies_collaborative"

    # --- Streamlit Page Configuration ---
    st.title("🎬 Movie Recommendation System")
    st.write("Find movies you'll love based on your taste!")

    # Create tabs for different recommendation methods
    tab1, tab2 = st.tabs(["Collaborative Filtering", "Content-Based"])

    with tab1:
        st.subheader("Collaborative Filtering")
        st.caption("Finds users with similar tastes and recommends movies they liked.")

        # --- Data Loading (Cached) ---
        @st.cache_data
        def load_data():
            try:
                ratings_df = pd.read_csv('dataset/ratings.csv', low_memory=False)
                movies_df = pd.read_csv('dataset/movies.csv', low_memory=False)
                links_df = pd.read_csv('dataset/links.csv')

                ratings_df['movieId'] = ratings_df['movieId'].astype(str)
                movies_df['movieId'] = movies_df['movieId'].astype(str)
                links_df['movieId'] = links_df['movieId'].astype(str)  # Ensure consistent string type
                links_df['imdbId'] = 'tt' + links_df['imdbId'].astype(str).str.zfill(7)

                # Create title to ID mapping for easier selection
                movie_title_to_id = movies_df.set_index('title')['movieId'].to_dict()
                movie_id_to_title = movies_df.set_index('movieId')['title'].to_dict()

                return ratings_df, movies_df, links_df, movie_title_to_id, movie_id_to_title
            except FileNotFoundError:
                st.error("Data files (ratings.csv, movies.csv, links.csv) not found in the 'dataset/' directory.")
                st.stop()
            except Exception as e:
                st.error(f"An error occurred during data loading: {e}")
                st.stop()

        # --- Qdrant Client Setup (Cached) ---
        @st.cache_resource
        def get_qdrant_client():
            client = QdrantClient(":memory:")  # Use memory for demo purposes
            return client

        @st.cache_resource
        def setup_qdrant_collection(_client, _ratings_df):
            try:
                _client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={},
                    sparse_vectors_config={
                        "ratings": models.SparseVectorParams()
                    }
                )

                # Aggregate ratings (mean rating per user per movie)
                ratings_agg_df = _ratings_df.groupby(['userId', 'movieId']).rating.mean().reset_index()

                user_sparse_vectors = defaultdict(lambda: {"values": [], "indices": []})
                for row in ratings_agg_df.itertuples():
                    try:
                        movie_id_int = int(row.movieId)
                        user_sparse_vectors[row.userId]["values"].append(float(row.rating))
                        user_sparse_vectors[row.userId]["indices"].append(movie_id_int)
                    except ValueError:
                        continue

                points_to_upload = []
                for user_id, sparse_vector in user_sparse_vectors.items():
                    if sparse_vector["indices"] and sparse_vector["values"]:
                        points_to_upload.append(PointStruct(
                            id=user_id,
                            vector={"ratings": SparseVector(
                                indices=sparse_vector["indices"],
                                values=sparse_vector["values"]
                            )},
                            payload={"user_id": user_id}
                        ))

                if points_to_upload:
                    _client.upload_points(
                        collection_name=collection_name,
                        points=points_to_upload,
                        wait=True
                    )
                    return True
                else:
                    st.warning("No valid user rating vectors were generated to upload.")
                    return False
            except Exception as e:
                st.error(f"Error setting up Qdrant collection: {e}")
                return False

        # --- Helper Functions ---
        def get_movie_poster(imdb_id, api_key):
            url = f"https://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data.get("Response") == "True":
                    return data.get('Poster', None), data.get('Title', 'N/A'), data.get('Plot', 'N/A'), data.get('imdbID')
                else:
                    return None, 'N/A', 'N/A', None
            except requests.exceptions.RequestException:
                return None, 'N/A', 'N/A', None

        def to_vector(ratings):
            vector = SparseVector(values=[], indices=[])
            for movie_id, rating in ratings.items():
                try:
                    vector.values.append(float(rating))
                    vector.indices.append(int(movie_id))
                except ValueError:
                    continue
            return vector

        # --- Recommendation Generation Function ---
        def generate_recommendations(user_ratings, _qdrant_client, _ratings_df, _movies_df, _links_df, _movie_id_to_title):
            if not user_ratings:
                st.info("Please rate at least one movie to get recommendations.")
                return

            st.subheader("Your Ratings")
            rated_cols = st.columns(len(user_ratings))
            i = 0
            for movie_id, rating in user_ratings.items():
                title = _movie_id_to_title.get(str(movie_id), f"Movie ID {movie_id}")
                with rated_cols[i]:
                    st.caption(title)
                    st.write(f"⭐ {rating}")
                i += 1

            # Convert user ratings to sparse vector
            user_vector = to_vector(user_ratings)

            if not user_vector.indices:
                st.warning("No valid ratings provided to generate recommendations.")
                return

            # Search Qdrant for similar users
            try:
                results = _qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=NamedSparseVector(
                        name="ratings",
                        vector=user_vector
                    ),
                    limit=20
                )
            except Exception as e:
                st.error(f"Error searching Qdrant: {e}")
                return

            similar_user_ids = [hit.id for hit in results]

            if not similar_user_ids:
                st.warning("Could not find users with similar ratings based on your input.")
                return

            # Get ratings from similar users
            similar_ratings_df = _ratings_df[_ratings_df['userId'].isin(similar_user_ids)]

            # Filter out movies the user has already rated
            movies_to_exclude = set(user_ratings.keys())
            potential_recs = similar_ratings_df[~similar_ratings_df['movieId'].astype(int).isin(movies_to_exclude)]

            if potential_recs.empty:
                st.info("Found similar users, but they haven't rated any movies you haven't seen.")
                return

            # Calculate average rating for potential recommendations by similar users
            rec_scores = potential_recs.groupby('movieId')['rating'].mean().sort_values(ascending=False)

            st.subheader("Recommended For You")

            # --- Display Results ---
            top_n = 10
            cols = st.columns(5)  # Create 5 columns for layout
            col_idx = 0  # Fix: Initialize as integer, not float

            # Merge with movie titles and links
            top_recs_df = rec_scores.head(top_n * 2)  # Get more initially to account for missing posters/links
            top_recs_df = top_recs_df.reset_index()
            top_recs_df['movieId'] = top_recs_df['movieId'].astype(str)
            top_recs_with_info = top_recs_df.merge(
                _movies_df[['movieId', 'title']], on='movieId', how='left'
            ).merge(
                _links_df[['movieId', 'imdbId']], on='movieId', how='left'
            )

            displayed_count = 0
            for idx, row in top_recs_with_info.iterrows():
                if displayed_count >= top_n:
                    break

                movie_id = row['movieId']
                score = row['rating']  # This is the average rating from similar users
                title = row['title']
                imdb_id = row['imdbId']

                if pd.isna(imdb_id) or pd.isna(title):
                    continue

                poster_url, api_title, plot, imdb_id = get_movie_poster(imdb_id, omdb_api_key)

                with cols[col_idx]:
                    if poster_url and poster_url != 'N/A':
                        # Make the image clickable
                        imdb_link = f"https://www.imdb.com/title/{imdb_id}/"
                        st.markdown(f"<a href='{imdb_link}' target='_blank'><img src='{poster_url}' width='100%'></a>", 
                                   unsafe_allow_html=True)
                        st.caption(f"Avg. Rating: {score:.2f}")
                    else:
                        st.caption("No Poster")
                    
                    # Make the title clickable
                    if imdb_id:
                        imdb_link = f"https://www.imdb.com/title/{imdb_id}/"
                        st.markdown(f"<a href='{imdb_link}' target='_blank'><b>{api_title or title}</b></a>", 
                                   unsafe_allow_html=True)
                    else:
                        st.write(f"**{api_title or title}**")

                col_idx = (col_idx + 1) % 5  # Move to the next column, wrap around
                displayed_count += 1

            if displayed_count == 0:
                st.info("No recommendations found based on the criteria and available data.")

        # Load data
        with st.spinner("Loading movie data..."):
            ratings_df, movies_df, links_df, movie_title_to_id, movie_id_to_title = load_data()

        # Setup Qdrant
        with st.spinner("Setting up recommendation engine..."):
            qdrant_client = get_qdrant_client()
            setup_success = setup_qdrant_collection(qdrant_client, ratings_df)

        if not setup_success:
            st.error("Failed to setup the recommendation engine. Please try again later.")
            return

        # Get movie titles for selector
        movie_titles = sorted(list(movie_title_to_id.keys()))

        # User rating form
        with st.form(key='rating_form'):
            st.write("Rate movies to get personalized recommendations:")
            selected_titles = st.multiselect(
                label="Select movies you've seen:",
                options=movie_titles,
                max_selections=5,
                placeholder="Choose up to 5 movies"
            )

            user_ratings = {}
            if selected_titles:
                st.write("Rate the selected movies (1=Terrible, 5=Excellent):")
                rating_cols = st.columns(min(3, len(selected_titles)))
                col_idx = 0
                for title in selected_titles:
                    movie_id = movie_title_to_id.get(title)
                    if movie_id:
                        with rating_cols[col_idx % min(3, len(selected_titles))]:
                            rating = st.slider(f"{title}", 1, 5, 3, key=f"rating_{movie_id}")
                            user_ratings[int(movie_id)] = rating
                        col_idx += 1

            submitted = st.form_submit_button("Get Recommendations")

        # Generate and display recommendations
        if submitted and user_ratings:
            with st.spinner("Finding movies you'll love..."):
                st.divider()
                generate_recommendations(user_ratings, qdrant_client, ratings_df, movies_df, links_df, movie_id_to_title)
        elif submitted and not user_ratings:
            st.warning("Please select and rate at least one movie.")
        else:
            st.info("Select and rate up to 5 movies above, then click 'Get Recommendations'.")

    with tab2:
        st.subheader("Content-Based Recommendation")
        st.caption("Recommends movies similar to ones you already like based on genres, actors, directors, etc.")
        
        st.info("Select a movie to find others with similar content:")
        
        # Simple content-based search
        selected_movie = st.selectbox("Choose a movie:", 
                                    options=movie_titles,
                                    index=None,
                                    placeholder="Select a movie...")
        
        if selected_movie and st.button("Find Similar Movies"):
            st.spinner("Analyzing movie content...")
            
            # Get the selected movie's ID and genre information
            movie_id = movie_title_to_id.get(selected_movie)
            if movie_id:
                # Find the genres for this movie
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    genres = movie_info.iloc[0]['genres'].split('|')
                    
                    # Find movies with similar genres
                    similar_movies = []
                    for idx, row in movies_df.iterrows():
                        row_genres = row['genres'].split('|')
                        # Calculate genre overlap
                        common_genres = set(genres) & set(row_genres)
                        if common_genres and row['movieId'] != movie_id:
                            similar_movies.append({
                                'movieId': row['movieId'],
                                'title': row['title'],
                                'genres': row['genres'],
                                'similarity': len(common_genres) / len(set(genres + row_genres))  # Jaccard similarity
                            })
                    
                    # Sort by similarity
                    similar_movies = sorted(similar_movies, key=lambda x: x['similarity'], reverse=True)[:10]
                    
                    if similar_movies:
                        st.subheader(f"Movies similar to '{selected_movie}'")
                        
                        # Display in a grid
                        cols = st.columns(5)
                        col_idx = 0
                        
                        for movie in similar_movies:
                            imdb_link = None
                            # Try to get the IMDB ID
                            movie_links = links_df[links_df['movieId'] == movie['movieId']]
                            if not movie_links.empty:
                                imdb_id = movie_links.iloc[0]['imdbId']
                                poster_url, api_title, plot, imdb_id = get_movie_poster(imdb_id, omdb_api_key)
                                
                                with cols[col_idx % 5]:
                                    if poster_url and poster_url != 'N/A':
                                        imdb_link = f"https://www.imdb.com/title/{imdb_id}/"
                                        st.markdown(f"<a href='{imdb_link}' target='_blank'><img src='{poster_url}' width='100%'></a>", 
                                                   unsafe_allow_html=True)
                                    else:
                                        st.caption("No Poster")
                                    
                                    # Make the title clickable
                                    if imdb_id:
                                        imdb_link = f"https://www.imdb.com/title/{imdb_id}/"
                                        st.markdown(f"<a href='{imdb_link}' target='_blank'><b>{api_title or movie['title']}</b></a>", 
                                                  unsafe_allow_html=True)
                                    else:
                                        st.write(f"**{movie['title']}**")
                                    
                                    st.caption(f"Similarity: {movie['similarity']:.2f}")
                                    st.caption(f"Genres: {movie['genres']}")
                            
                            col_idx += 1
                    else:
                        st.info("No similar movies found.")
            else:
                st.error("Movie ID not found.")
