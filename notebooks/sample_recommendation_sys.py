import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# dataset = pd.read_csv('Recommendation_System/notebooks/toy_data.csv')

# dataFrame_dt = pd.DataFrame(dataset)

# movie_overview = dataFrame_dt['overview']

# vectorizer = TfidfVectorizer(stop_words='english')

# tfidf_matrix = vectorizer.fit_transform(movie_overview)

# print(dataFrame_dt)
# print(dataFrame_dt.shape)

# print(dataFrame_dt['overview'])
# print("Feature Names: ", vectorizer.get_feature_names_out())

# print("Arryay:", tfidf_matrix.toarray())
# print(tfidf_matrix.shape)

# interstellar_vector = tfidf_matrix[0].toarray().flatten(
# )

# print(interstellar_vector)

# feature_names = vectorizer.get_feature_names_out()

# interstellar_features = list(
#     zip(feature_names, interstellar_vector)
# )

# interstellar_features = sorted(
#     interstellar_features,
#     key= lambda x: x[1],
#     reverse= True
# )

# for word,score in interstellar_features:
#     if score > 0:
#         print(f'{word:15} {score:.4f}')

# similarity_matrix = cosine_similarity(tfidf_matrix)

# print(similarity_matrix.shape)

# interstellar_movies = dataFrame_dt[
#     dataFrame_dt['title'] == 'Interstellar'
# ].index[0]

# print(similarity_matrix[interstellar_movies])



# def recommend(movie_name,top_k):

#     try: 
#         movie_index = dataFrame_dt[dataFrame_dt['title'] == movie_name].index[0] 

#         similarity_scores = similarity_matrix[movie_index]

#         val = []

#         for movie,score in zip(dataFrame_dt['title'],similarity_scores):

#             if movie != movie_name:
#                 val.append((movie,float(score)))

#         sorted_dict_val = sorted(val,key=lambda x:x[1],reverse= True)

#         print(sorted_dict_val[:top_k])

#     except (ValueError):
#         print('Unable to Execute the code due to Value error.')
#     except (IndexError):
#         print('Unable to Execute the code due to Index error.')


# recommend("Interstellar",3)