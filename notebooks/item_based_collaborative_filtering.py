
import pandas as pd
import math

def item_similarity(item_a, item_b):

    # 1. Find movies both users have rated.
    common_movies = item_a.notna() & item_b.notna()
    length = sum(1 for item in common_movies if item is True)

    if length == 0:
        return 0.0, 0

    # return common_movies
    # 2. Extract their ratings for those movies.

    ratings_a = item_a[common_movies]
    ratings_b = item_b[common_movies]

    # 3. Calculate the dot product.
    def dot_products(a,b):

        dot_product = 0

        for x, y in zip(a,b):
            dot_product += ( x * y)

        return dot_product
    
    # 4. Calculate both vector magnitudes.

    def magnitude(values):

        magnitude = 0

        for value in values:

            magnitude += (value) ** 2

        return math.sqrt(magnitude)

    
    
    # 5. Handle a zero magnitude safely.
    
    # 6. Return cosine similarity.
    def cosine_similarities(a,b):

        dot = dot_products(a,b)

        mag_a = magnitude(a)

        mag_b = magnitude(b)

        denominator = mag_a*mag_b

        if denominator == 0:
            return 0.0

        cosine_similarity = dot/denominator

        return cosine_similarity

    return cosine_similarities(ratings_a,ratings_b), length


ratings = [
    ["U1", "Interstellar", 5],
    ["U1", "The Martian", 4],
    ["U1", "Gravity", 5],
    ["U1", "Titanic", 1],

    ["U2", "Interstellar", 4],
    ["U2", "The Martian", 5],
    ["U2", "Gravity", 4],
    ["U2", "Titanic", 1],

    ["U3", "Interstellar", 1],
    ["U3", "The Martian", 2],
    ["U3", "Gravity", 1],
    ["U3", "Titanic", 5],
    ["U3", "Avengers", 4],

    ["U4", "Interstellar", 2],
    ["U4", "The Martian", 1],
    ["U4", "Gravity", 2],
    ["U4", "Titanic", 4],
    ["U4", "Avengers", 5],

    ["U5", "Interstellar", 5],
    ["U5", "Gravity", 4],
    ["U5", "Avengers", 2],
]

ratings_df = pd.DataFrame(
    ratings,
    columns=["user_id", "movie_title", "rating"]
)

user_item_matrix = ratings_df.pivot(index='user_id',columns='movie_title',values = 'rating')


movie_names = user_item_matrix.columns

item_similarity_matrix = pd.DataFrame(index=movie_names,columns=movie_names)

def similarity_matrix(movie_names, matrix):
    similarity_matrix = pd.DataFrame(
        index=movie_names,
        columns=movie_names
    )

    for row_name in movie_names:
        for col_name in movie_names:
            value = item_similarity(
                matrix[row_name],
                matrix[col_name]
            )[0]

            similarity_matrix.loc[row_name, col_name] = value

    return similarity_matrix

item_similarity_matrix = similarity_matrix(movie_names,user_item_matrix)


def find_similar_movies(movie_title, item_similarity_matrix, top_k=3):

    #     Check whether movie_title exists in the matrix; raise a ValueError if it doesn't.
    if movie_title not in item_similarity_matrix:
        raise ValueError(f"{movie_title} is doesn't exist in the matrix.")
    
    # Select that movie's similarity scores.
    def most_similar_user(movie_title, item_similarity_matrix):

        similar_user = {}
        
        for movie in item_similarity_matrix.columns:            
            
            if movie != movie_title:

                similarity = item_similarity_matrix.loc[movie_title,movie]

                similar_user[movie] = similarity

        sorted_similar_user = dict(sorted(similar_user.items(),key=lambda x:x[1],reverse= True))

        return sorted_similar_user

    value =  most_similar_user(movie_title, item_similarity_matrix)

    return value



def predict_item_rating(target_user, target_movie, user_item_matrix, item_similarity_matrix):

    # Retrieve the target user's ratings.

    user_rating =  user_item_matrix.loc[target_user]

    # Keep only movies they've rated, excluding the target movie.
    seen_movies = user_rating[user_rating.notna()].index

    # Get each rated movie's similarity to the target movie.


    numerator = 0.0
    denominator = 0.0
    
    for movie in seen_movies:

        similarity = item_similarity_matrix.loc[target_movie,movie]

        rating = user_rating[movie]

        numerator += (similarity * rating)

        denominator += abs(similarity)

    if denominator == 0: 
        return None

    return numerator/denominator

def recommend_items(target_user, user_item_matrix,item_similarity_matrix, top_k=3):
    # Find movies the target user hasn't rated.
    user_row = user_item_matrix.loc[target_user]
    unseen_movies = user_row[user_row.isna()].index

    predictions = {}

    # Predict each unseen movie's rating using predict_item_rating().
    for movie in unseen_movies:
        predicted_rating = predict_item_rating(target_user,movie,user_item_matrix,item_similarity_matrix)
        # Skip movies without a valid prediction.
        if predicted_rating is not None:
            predictions[movie] = predicted_rating
    
    # Sort by predicted rating, highest first.
    ranked = sorted(
        predictions.items(),
        key= lambda item: item[1],
        reverse=True
    )

    # Return the top top_k movies.
    return ranked[:top_k]


target_user = "U1"

movie_title = "Avengers"
    
result = find_similar_movies(movie_title, item_similarity_matrix, top_k=3)

print(result)


# def calculate_metrics(actual, predicted):
#     if len(actual) == 0 or len(actual) != len(predicted):
#         raise ValueError("Inputs must have equal, non-zero lengths.")

#     n = len(actual)

#     absolute_error = 0.0
#     squared_error = 0.0

#     for act, pred in zip(actual, predicted):
#         error = act - pred
#         absolute_error += abs(error)
#         squared_error += error ** 2

#     mae = absolute_error / n
#     rmse = math.sqrt(squared_error / n)

#     return {"MAE": mae, "RMSE": rmse}



# actual = [1]
# user_predictions = [3.2920092921870943]
# item_predictions = [4.6814099666323346]

# print(calculate_metrics(actual,user_predictions))
# print(calculate_metrics(actual,item_predictions))

# def evaluate_single_holdout(test_user, test_movie):
#     # Gets the actual rating from ratings_df.

#     test_rating = ratings_df[
#         (ratings_df["user_id"] == test_user) &
#         (ratings_df["movie_title"] == test_movie)
#     ]["rating"].iloc[0]

#     # Removes that rating to create train_ratings_df.

#     train_ratings_df = ratings_df[
#         ~(
#             (ratings_df["user_id"] == test_user) &
#             (ratings_df["movie_title"] == test_movie)
#         )
#     ].copy()


#     # Builds train_user_item_matrix.
#     train_user_item_matrix = train_ratings_df.pivot(
#         index='user_id',
#         columns='movie_title',
#         values='rating'
#     )

#     # Builds both training-only similarity matrices.

#     movie_names = train_user_item_matrix.columns


#     test_item_similarity_matrix = similarity_matrix(
#         movie_names,
#         train_user_item_matrix
#     )

#     item_prediction = predict_item_rating(
#         test_user,
#         test_movie,
#         train_user_item_matrix,
#         test_item_similarity_matrix
#     )

#     return {
#         "Actual": test_rating,
#         "Item Prediction": item_prediction
#     }




#     # Returns the actual rating and both predictions
# # test_user = "U1"
# # test_movie = "Titanic"

# # print(evaluate_single_holdout(test_user,test_movie))
# print(evaluate_single_holdout("U1", "Titanic"))
# print(evaluate_single_holdout("U3", "Avengers"))