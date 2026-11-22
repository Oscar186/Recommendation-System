from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import WordNetLemmatizer
movies_df = pd.read_csv('Recommendation_System/notebooks/movie_recom.csv')

lemmatizer = WordNetLemmatizer()

movies_df["combined_features"] = (
    movies_df['genre'] + " "+
    movies_df['director'] + " "+
    movies_df['cast']+ " "+
    movies_df["keywords"] + " "+
    movies_df['overview']
)

def lemmatize_text(text):

    words_list = text.split(' ')

    new_text = ''
    
    for word in words_list:
        new_text += lemmatizer.lemmatize(word,pos="v") + ' '

    return new_text


movies_df['overview_lemmatized'] = movies_df['overview'].apply(lemmatize_text)

# print(movies_df.iloc[:,6:])

# print(movies_df[['title','overview','overview_lemmatized']])

# print('-----------------------------------------------------------------------------------------------------------------')

original_vectorizer = TfidfVectorizer(stop_words='english',)
original_matrix = original_vectorizer.fit_transform(movies_df['combined_features'])

lemmatized_vectorixer = TfidfVectorizer(stop_words='english')
lemmatized_matrix = lemmatized_vectorixer.fit_transform(movies_df['overview_lemmatized'])

# print("Original Matrix: ",original_matrix.shape)
# print("Lemmatized Matrix: ",lemmatized_matrix.shape)

original_features = original_vectorizer.get_feature_names_out()
lemmatized_features = lemmatized_vectorixer.get_feature_names_out()
# print("----------------------------------------------------------------------------------------------------------------")
# print("Original Features: ", original_features)
# print("Lemmatized Features: ", lemmatized_features)

originial_similarity = cosine_similarity(original_matrix)
lemmatized_similarity = cosine_similarity(lemmatized_matrix)

# print("Original")
# print(original_matrix[0])

# print("Lemmatized")
# print(lemmatized_matrix[0])

print("----------------------------------------------------------------------------------------------------------------")


# def recommend(movie_name,top_k):

#     # try: 
#         movie_index = movies_df[movies_df['title'] == movie_name].index[0] 

#         original_similarity_scores = originial_similarity[movie_index]
#         lemmatized_similarity_scores = lemmatized_similarity[movie_index]

#         original_val,lemmatized_val = [],[]

#         def predictor(val,similarity_score):

#             for movie,score in zip(movies_df['title'],similarity_score):

#                 if movie != movie_name:
#                     val.append((movie,float(score)))

#             sorted_dict_val = sorted(val,key=lambda x:x[1],reverse= True)

#             return sorted_dict_val[:top_k]
        
#         print("Original Recommendation") 
#         print(predictor(original_val,original_similarity_scores))

#         print("Lemmatized Recommedation")
#         print(predictor(lemmatized_val,lemmatized_similarity_scores))

            # except IndexError:
            #     return f"Movie '{movie_name}' not found."

            # except ValueError:
            #     return "Unable to execute recommendation."
        

# recommend("The Martian",3)

def recommend(movie_name,top_k):


    if movie_name not in movies_df['title'].values:
        raise ValueError(f'Movie {movie_name} not found.')
    
    movie_index = movies_df[movies_df['title'] == movie_name].index[0] 

    # lemmatized_similarity_scores = lemmatized_similarity[movie_index]
    original_similarity_scores = originial_similarity[movie_index]
    # similarity_scores = similarity_matrix[movie_index]

    recommendation = []

    for index,score in enumerate(original_similarity_scores):

        if index != movie_index:

            movie = movies_df['title'].iloc[index]

            recommendation.append((movie,float(score)))

    sorted_dict_val = sorted(recommendation,key=lambda x:x[1],reverse= True)

    return sorted_dict_val[:top_k]




print(recommend("Interstellar",5))

print(recommend('Avatar',5))

# print(movies_df[["title",'genre','director','cast','keywords']])
# print('.........................................................................................')
# for movie in movies_df['director']:
#     print(movie)
# print('.........................................................................................')
# for value in movies_df['cast']:
#     print(value)

# def normalize_entity(text):
#     return text.lower().replace(" ","_")

# print(normalize_entity("Christopher Nolan"))
# print(normalize_entity("Quentin Tarantino"))
# print(normalize_entity("Robert Downey Jr"))