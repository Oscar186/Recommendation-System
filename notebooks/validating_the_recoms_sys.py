import pandas as pd
import numpy as np

actual = np.array([5, 4, 3, 2, 5])

predicted = np.array([4, 3.5, 4, 2.5, 4])

absolute_error = np.abs(actual-predicted)

mean_absolute_error = np.mean(absolute_error)

squared_error = (actual-predicted)**2

root_mean_squared_error = np.sqrt(np.mean(squared_error))

ratings_df = pd.DataFrame({
    "user_id": [
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4
    ],
    "movie_title": [
        "Inception", "Titanic", "Interstellar",
        "The Martian", "Avatar",

        "Inception", "Titanic", "Interstellar",
        "The Martian", "Avatar",

        "Inception", "Titanic", "Interstellar",
        "The Martian", "Avatar",

        "Inception", "Titanic", "Interstellar",
        "The Martian", "Avatar"
    ],
    "rating": [
        5, 4, 5, 4, np.nan,
        4, 5, 4, 5, 3,
        2, 3, 5, 4, 4,
        3, 2, 4, 5, 4
    ]
})

ratings_df.groupby('user_id').size()


known_ratings_df = ratings_df.dropna(
    subset=['rating']
)

missing_ratings_df = ratings_df[ratings_df['rating'].isna()]


test_df = known_ratings_df.groupby(
    "user_id",
    group_keys=False,
).sample(n=1,random_state=42)

train_df = known_ratings_df.drop(test_df.index)

# print("Missing Data Frame: ")
# print(missing_ratings_df)

# print("Training data:")
# print(train_df)

# print("\nTesting data:")
# print(test_df)

train_matrix = train_df.pivot(
    index='user_id',
    columns='movie_title',
    values='rating'
)

# print(train_matrix)

user_mean = train_matrix.mean(axis=1)

centered_matrix = train_matrix.sub(
    user_mean,
    axis=0
)

# print("User Means:")
# print(user_mean)

# print('Centered Matrix: ')
# print(centered_matrix)

filled_matrix = np.nan_to_num(
    centered_matrix.to_numpy()
)

# print('Filled Matrix: ')
# print(filled_matrix)

U, Sigma, Vt = np.linalg.svd(
    filled_matrix,
    full_matrices= False
)


K = 4

U_k = U[:, :K]

Sigma_k = np.diag(Sigma[:K])

VT_k = Vt[:K, :]

reconstructed_matrix = U_k @ Sigma_k @ VT_k

print("U Shape:", U_k.shape)
print("Sigma:", Sigma_k.shape)
print("VT Shape:", VT_k.shape)

print("Reconstruction error:",
      np.linalg.norm(filled_matrix - reconstructed_matrix))

predicted_matrix = (
    reconstructed_matrix + user_mean.to_numpy().reshape(-1,1)
)

print("Predicted Rating Matrix:")
print(predicted_matrix)

predictions = []

for _, row in test_df.iterrows():

    user_idx = train_matrix.index.get_loc(
        row["user_id"]
    )

    movie_idx = train_matrix.columns.get_loc(
        row["movie_title"]
    )

    predicted_rating = predicted_matrix[
        user_idx, movie_idx
    ]

    predictions.append(predicted_rating)

test_df = test_df.copy()

test_df["predicted_rating"] = predictions

print("Test Data Frame: ")
print(test_df)


actual = test_df["rating"].to_numpy()

predicted = test_df["predicted_rating"].to_numpy()

mae = np.mean(np.abs(actual - predicted))

rmse = np.sqrt(np.mean((actual - predicted) ** 2))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")