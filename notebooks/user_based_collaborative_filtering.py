
import pandas as pd
import math

def user_similarity(user_a, user_b):

    # 1. Find movies both users have rated.
    common_movies = user_a.notna() & user_b.notna()
    length = sum(1 for item in common_movies if item is True)

    if length == 0:
        return 0.0, 0

    # return common_movies
    # 2. Extract their ratings for those movies.

    ratings_a = user_a[common_movies]
    ratings_b = user_b[common_movies]

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

users = ['U1','U2','U3','U4','U5']


def most_similar_user(target_user, users):

        similar_user = {}
        
        for user in users:            
            
            if user != target_user:

                similarity = user_similarity(user_item_matrix.loc[target_user],user_item_matrix.loc[user])

                similar_user[user] = [similarity[0],similarity[1]]

        sorted_similar_user = dict(sorted(similar_user.items(),key=lambda x:x[1],reverse= True))

        return sorted_similar_user


def recommed_movies(target_user,user_item_matrix,result,top_k = 3):

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    
    # 1. identify the user hasn't rated
    user_row = user_item_matrix.loc[target_user]
    unseen_movies = user_row[user_row.isna()].index

    predictions = {}


    for movie in unseen_movies:
        numerator = 0.0
        denominator = 0.0

        for neighbor, (similarity,count) in result.items():
            neighbor_rating = user_item_matrix.at[neighbor,movie]

            if pd.isna(neighbor_rating):
                continue

            numerator += similarity * neighbor_rating
            denominator += abs(similarity)

        if denominator == 0:
            continue

        predictions[movie] = numerator / denominator

    ranked = sorted(
        predictions.items(),
        key= lambda item: item[1],
        reverse=True
    )

    return ranked[:top_k]

target_user = "U1"
result = most_similar_user(target_user,users)
recommed_val = recommed_movies(target_user,user_item_matrix,result,3)

print(recommed_val)


test_user = "U1"
test_movie = "Titanic"

test_rating = ratings_df[
    (ratings_df["user_id"] == test_user) &
    (ratings_df["movie_title"] == test_movie)
]["rating"].iloc[0]

train_ratings_df = ratings_df[
    ~(
        (ratings_df["user_id"] == test_user) &
        (ratings_df["movie_title"] == test_movie)
    )
].copy()


def build_user_similarity_matrix(matrix):
    users = matrix.index

    similarity_matrix = pd.DataFrame(
        index=users,
        columns=users,
        dtype=float
    )

    for user_a in users:
        for user_b in users:
            similarity, count = user_similarity(
                matrix.loc[user_a],
                matrix.loc[user_b]
            )

            similarity_matrix.loc[user_a, user_b] = similarity

    return similarity_matrix


# result = most_similar_user(
#     "U1",
#     test_user_similarity_matrix
# )



# def predict_user_rating(target_user, target_movie, matrix, similarity_matrix):
#     numerator = 0.0
#     denominator = 0.0

#     for neighbor in matrix.index:
#         if neighbor == target_user:
#             continue

#         similarity = similarity_matrix.loc[target_user, neighbor]
#         neighbor_rating = matrix.loc[neighbor, target_movie]


#         if pd.isna(neighbor_rating):
#             continue

#         numerator += (similarity * neighbor_rating)

#         denominator += abs(similarity)

#     if denominator == 0.0:
#         return None

#     return numerator/denominator

# def evaluate_single_holdout(test_user, test_movie):
#     train_user_item_matrix = train_ratings_df.pivot(
#         index='user_id',
#         columns='movie_title',
#         values='rating'
#     )

#     test_user_similarity_matrix = build_user_similarity_matrix(
#         train_user_item_matrix
#     )

#     user_prediction = predict_user_rating(
#         test_user,
#         test_movie,
#         train_user_item_matrix,
#         test_user_similarity_matrix
#     )

#     return {
#         "Actual":test_rating,
#         "User Prediction" : user_prediction
#     }

# print(evaluate_single_holdout("U1", "Titanic"))
# print(evaluate_single_holdout("U3", "Avengers"))