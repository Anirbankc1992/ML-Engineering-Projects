try:
    from json import dumps
    import pandas as pd
    import numpy as np

    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.neighbors import NearestNeighbors

    from Recommenders.logger import file_logger
except ImportError as ie:
    print(f"error: {ie.__class__.__name__, ie.args[-1]}")
    
    

ingestion = folder = logger_task = "Recommendation"
recommend_logger = file_logger(file_function=ingestion, logger_task=logger_task, folder=folder)

class MovieAnalyzer:
    """
    Class for analyzing movie data, creating aggregations.
    """

    def __init__(self, ratings_file: str, movies_file: str):
        """
        Initialize MovieAnalyzer with ratings and movies files.

        Args:
            ratings_file (str): Path to the ratings file.
            movies_file (str): Path to the movies file.
        """
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.df = None

    def load_data(self):
        """Load ratings and movies data."""
        self.ratings = pd.read_csv(self.ratings_file)
        self.movies = pd.read_csv(self.movies_file)
        self.df = pd.merge(self.ratings, self.movies, on='movieId')

    def preprocess_genres(self):
        """Preprocess genres data."""
        self.df['genres'] = self.df['genres'].str.split('|')
        self.df['genres'] = self.df['genres'].fillna("").astype('str')

    def get_average_ratings(self) -> pd.Series:
        """Get average ratings for each movie."""
        average_ratings = self.df.groupby('title')['rating'].mean().sort_values(ascending=False)
        return average_ratings

    def get_vote_counts(self) -> pd.Series:
        """Get vote counts for each movie."""
        vote_counts = self.df.groupby('title')['rating'].count().sort_values(ascending=False)
        return vote_counts

    def get_ratings_with_count(self) -> pd.DataFrame:
        """Get ratings with count for each movie."""
        ratings = self.df.groupby('title')['rating'].mean()
        count = self.df.groupby('title')['rating'].count()
        ratings_df = pd.DataFrame(ratings)
        ratings_df['count'] = count
        return ratings_df
    
    def __call__(self):
        """Call method to load data, preprocess, and return analysis results."""
        self.load_data()
        self.preprocess_genres()
        average_ratings = self.get_average_ratings()
        vote_counts = self.get_vote_counts()
        ratings_with_count = self.get_ratings_with_count()
        return average_ratings.reset_index(), vote_counts.reset_index(), ratings_with_count.reset_index()
        

class TitleRecommender:
    """
    Class for recommending movies based on titles.
    """

    def __init__(self, movies_df: pd.DataFrame, rows: int):
        """
        Initialize TitleRecommender with movie data and number of rows.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie data.
            rows (int): Number of rows to consider.
        """
        self.movies_df = movies_df.head(rows)
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self._prepare_data()

    def _prepare_data(self, metric='cosine'):
        """
        Prepare data for recommendation.

        Args:
            metric (str, optional): Similarity metric to use. Defaults to 'cosine'.
        """
        try:
            tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.2, stop_words='english')
            self.tfidf_matrix = csr_matrix(tfidf.fit_transform(self.movies_df['genres']))
        
            # Compute cosine similarity using pairwise_distances with cosine metric
            self.cosine_sim = 1 - pairwise_distances(self.tfidf_matrix, metric=metric)
        
            self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'])
        except Exception as e:
            error_msg = f"Error occured while calculating tdidf & cosine similarity {e.__class__.__name__, e.args[-1]}"
            print(error_msg)
            recommend_logger.critical(error_msg)

    def _compute_genre_similarity(self, target_genres: pd.Series, other_genres: pd.Series):
        """
        Compute genre similarity between two sets of genres.

        Args:
            target_genres (list): List of genres for target movie.
            other_genres (list): List of genres for another movie.

        Returns:
            float: Genre similarity score.
        """
        intersection = len(set(target_genres) & set(other_genres))
        union = len(set(target_genres) | set(other_genres))
        return intersection / union if union != 0 else 0

    def recommend_by_title(self, title: str, k:int) -> pd.DataFrame:
        """
        Recommend movies based on a given title.

        Args:
            title (str): Title of the movie.
            k (int): Top k recommendations given a title.

        Returns:
            pd.DataFrame: DataFrame containing recommended movies.
        """
        try:
            idx = self.indices[title]
            sim_scores = self.cosine_sim[idx]
        
             # Calculate genre similarity
            target_genres = self.movies_df.loc[idx, 'genres']
            genre_similarity = [self._compute_genre_similarity(target_genres, genres.split('|')) for genres in self.movies_df['genres']]
        
            # Combine cosine similarity and genre similarity
            final_scores = sim_scores + np.array(genre_similarity)
        
            sorted_indices = np.argsort(-final_scores)
        
            # Get similar movies
            similar_movies = self.movies_df.iloc[np.unique(sorted_indices)].sort_values(by='rating', ascending=False)
            similar_movies = similar_movies[~similar_movies['title'].str.contains(title)].drop_duplicates(subset=['title']).reset_index(drop=True)

        except Exception as e:
            error = f"Error occured while calculating title recommendation for {title}: {e.__class__.__name__, e.args[-1]}"
            print(error)
            recommend_logger.exception(error)
        else:
            similar_movies =  similar_movies[['movieId','title','genres']].head(k)
            if not similar_movies.empty:
                success = f"{len(similar_movies)} recommendations generated for {title} successfully!"
                print(success)
                recommend_logger.debug(success)
                return similar_movies
            else:
                zero_warn = f"No recommendations generated for {title}. Please check if title exists in the data source."
                print(zero_warn)
                recommend_logger.warning(zero_warn)    
    
class SimilarityUserRecommender:
    """
    Class for recommending movies to users based on similarity.
    """
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df
        self.user_similarity_matrix = None
        self._compute_similarity_matrices()

    def _compute_similarity_matrices(self,n_components=30,iterations=5,metric="cosine"):
        """Dimensionality reduction by Truncated Singular Value Decomposition
         of user-movie dataframe, comfollowed by cosine similarity computation by taking pairwise distances."""
        try:
            user_movie_rating = self.movies_df.pivot_table(index='userId', columns='title', values='rating') 
            user_movie_rating.fillna(0, inplace=True)

            svd = TruncatedSVD(n_components=n_components, n_iter=iterations, random_state=42)
            x = svd.fit_transform(user_movie_rating)
            user_movie_pred = svd.inverse_transform(x)
        
            dense_matrix = pd.DataFrame(user_movie_pred)

            user_similarity = 1 - pairwise_distances(dense_matrix, metric=metric)
            np.fill_diagonal(user_similarity, 0) 
            self.user_similarity_matrix = pd.DataFrame(user_similarity)
        except Exception as e:
            error = f"Error occured while calculating similarity matrix: {e.__class__.__name__, e.args[-1]}"
            print(error)
            recommend_logger.exception(error)
            

    def get_similar_users(self, user_id: str, metric='cosine', algorithm='auto', k=15) -> list:
        """
        Get similar users for a given user.

        Args:
            user_id (int): ID of the user.
            metric (str, optional): Similarity metric to use. Defaults to 'cosine'.
            algorithm (str, optional): Algorithm to use for nearest neighbors. Defaults to 'auto'.
            k (int, optional): Number of similar users to retrieve.

        Returns:
            list: List of tuples containing user_id and similarity score, separated by comma.
        """
        try:        
            similarities, indices = [],[]
            
            model_knn = NearestNeighbors(metric=metric, algorithm=algorithm) 
            model_knn.fit(self.user_similarity_matrix)
            distances, indices = model_knn.kneighbors(self.user_similarity_matrix.iloc[user_id - 1, :].values.reshape(1, -1), 
                                                      n_neighbors=k + 1)
            similarities = 1 - distances.flatten()

            similar_users = []
            for i in range(0, len(indices.flatten())):
                if indices.flatten()[i] + 1 == user_id:
                    continue
                else:
                    similar_user_id = indices.flatten()[i] + 1
                    similarity_score = similarities.flatten()[i]
                    
                    # Fetch movies of similar user
                    similar_user_movies = self.movies_df[self.movies_df['userId'] == similar_user_id] \
                                                                    [['title', 'rating', 'genres','timestamp']]
            
                    # If a user rated the same movie multiple times, average their ratings
                    user_ratings = similar_user_movies.groupby('title').agg({'rating': 'mean',
                                                                            'timestamp': 'max'}).reset_index()
        
                    # Calculate top 3 movies for similar user
                    top_movies = user_ratings.sort_values(by=['rating', 'timestamp'], 
                                                          ascending=[False, False]).head(3).reset_index(drop=True)
        
                    similar_users.append(dict(user_id = similar_user_id, 
                                             similarity_score = similarity_score,
                                             top_movies = top_movies.to_dict(orient='records'),
                                                ))
        except Exception as e:
            error = f"Error occured while calculating title recommendation for {user_id}: {e.__class__.__name__, e.args[-1]}"
            print(error)
            recommend_logger.exception(error)
        else:
            recommended_movies = pd.DataFrame(similar_users) 
            if not recommended_movies.empty:
                success = f"{len(recommended_movies)} recommendations generated successfully for {user_id}"
                print(success)
                recommend_logger.debug(success)
                return recommended_movies
            else:
                zero_warn = f"No recommendations generated for {user_id}. Please check if user_id exists in the data source."
                print(zero_warn)
                recommend_logger.warning(zero_warn)