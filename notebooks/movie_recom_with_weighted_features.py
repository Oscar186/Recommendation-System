from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

movies_df = pd.read_csv('Recommendation_System/notebooks/movie_recom.csv')


movies_df["combined_features"] = (
    (movies_df['genre'] + " ") * 3+
    movies_df['director'] + " "+
    movies_df['cast']+ " "+
    (movies_df["keywords"] + " ")* 2+
    movies_df['overview']
)


vectorizer = TfidfVectorizer(stop_words='english')

tfidfmatrix = vectorizer.fit_transform(movies_df['combined_features'])

similarity_matrix = cosine_similarity(tfidfmatrix)

# print(tfidfmatrix.shape)
# print(similarity_matrix.shape)

def recommend(movie_name,top_k):

    try: 
        movie_index = movies_df[movies_df['title'] == movie_name].index[0] 

        similarity_scores = similarity_matrix[movie_index]

        recommendation = []

        for index,score in enumerate(similarity_scores):

            if index != movie_index:
                movie = movies_df['title'].iloc[index]
                recommendation.append((movie,float(score)))

        sorted_dict_val = sorted(recommendation,key=lambda x:x[1],reverse= True)

        return sorted_dict_val[:top_k]


    except ValueError:
        return f"Movie '{movie_name} not found'"
