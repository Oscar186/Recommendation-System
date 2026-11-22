import pandas as pd

ratings_df = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 4],
    "movie_title": [
        "Interstellar",
        "Titanic",
        "Interstellar",
        "The Martian",
        "Titanic",
        "Interstellar",
        "The Martian"
    ],
    "rating": [5, 4, 5, 4, 5, 4, 5]
})


def recommend_popular_movies(ratings_df,top_k,user_id):

    popularity_df = (ratings_df.groupby('movie_title').size().reset_index(name="ratings_count"))

    popularity_df = popularity_df.sort_values(by= "ratings_count",ascending=False)

    user_movies = ratings_df[ratings_df["user_id"] == user_id]['movie_title']

    movies_by_user = popularity_df[~popularity_df['movie_title'].isin(user_movies)]

    return movies_by_user[:top_k]

def recommend_weighted_movies(ratings_df, user_id, top_k):
    # Calculate Average Rating
    average_rating_df = (ratings_df.groupby('movie_title')['rating'].mean().reset_index(name = 'average_rating'))
    average_rating_df = average_rating_df.sort_values(by='average_rating',ascending = False)

    # Calculate Ratings Count
    total_ratings = ratings_df.groupby('movie_title').size().reset_index(name = 'ratings_count')

    # Calculate Global Average Rating
    global_avg = ratings_df['rating'].mean()

    # Manually giving threshold value
    threshold_value = 2

    new_df = pd.DataFrame(average_rating_df)

    new_df1 = pd.DataFrame(total_ratings)

    modified_df = pd.merge(new_df,new_df1,how='inner',on='movie_title')

    first_half = (modified_df['ratings_count']/(modified_df['ratings_count']+threshold_value))*modified_df['average_rating']

    second_half = (threshold_value/(modified_df['ratings_count']+threshold_value))*global_avg

    modified_df['weighted_avg'] = first_half+second_half

    modified_df.sort_values(by='weighted_avg',ascending=False)

    user_movies = ratings_df[ratings_df["user_id"] == user_id]['movie_title']

    movies_by_user = modified_df[~modified_df['movie_title'].isin(user_movies)]

    return movies_by_user[:top_k]


# print(recommend_weighted_movies(ratings_df,1,2))




