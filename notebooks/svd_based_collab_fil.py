
import pandas as pd
import numpy as np

# ratings_df = pd.DataFrame(data=[[5,4,np.nan],[4,np.nan,5],[np.nan,3,4]],index=['user1','user2','user3'],columns=['Interstellar','Titanic','Martian'])

# ratings_df

ratings_df = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "movie_title": [
        "Interstellar",
        "Titanic",
        "The Martian",
        "Interstellar",
        "Titanic",
        "The Martian",
        "Interstellar",
        "Titanic",
        "The Martian"
    ],
    "rating": [5, 4, np.nan,4,np.nan, 5, np.nan, 3, 4]
})

user_item_matrix = ratings_df.pivot_table(
    index="user_id",
    columns="movie_title",
    values="rating"
)

# print(user_item_matrix.head())
# print("Matrix shape:", user_item_matrix.shape)

total_entries = user_item_matrix.size
observed_ratings = user_item_matrix.notna().sum().sum()
missing_ratings = user_item_matrix.isna().sum().sum()

# print("Total entries:", total_entries)
# print("Observed ratings:", observed_ratings)
# print("Missing ratings:", missing_ratings)

sparsity = (missing_ratings/total_entries)*100

# print(f"Sparsity: {sparsity:.2f}%")

# Step 2: Calculate User Mean Ratings

user_means = user_item_matrix.mean(axis=1)

# Step 3: Mean Centering

centered_matrix = user_item_matrix.sub(user_means, axis=0)

# print(centered_matrix)

centered_matrix = centered_matrix.fillna(0)

# print(centered_matrix)

# print("User Mean Ratings:")
# print(user_means)

# columns = user_item_matrix.columns.to_list()

# for col in columns:
#     user_item_matrix[col] = user_item_matrix[col].fillna(user_item_matrix[col].mean(axis=1))

U,sigma, Vt = np.linalg.svd(
    centered_matrix.to_numpy(),
    full_matrices= False
)

# print("U Matrix:")
# print(U)

'''U Matrix:
[[-4.08248290e-01 -7.07106781e-01 -5.77350269e-01]
 [-4.08248290e-01  7.07106781e-01 -5.77350269e-01]
 [-8.16496581e-01  1.02457679e-16  5.77350269e-01]]'''

# print("\nSingular Values:")
# print(sigma)

'''Singular Values:
[8.66025404e-01 8.66025404e-01 1.15474807e-17]'''

# print("\nVT Matrix:")
# print(Vt)

'''VT Matrix:
[[-9.70141443e-17 -7.07106781e-01  7.07106781e-01]
 [-8.16496581e-01  4.08248290e-01  4.08248290e-01]
 [ 5.77350269e-01  5.77350269e-01  5.77350269e-01]]'''


# print("U Shape:", U.shape)
# Output: U Shape: (3, 3)
# print("Sigma Shape:", sigma.shape)
# Output: Sigma Shape: (3,)
# print("VT Shape:", Vt.shape)
# Output: VT Shape: (3, 3)

# Step 6: Calculate Explained Energy

singular_values_squared = sigma ** 2

total_energy = np.sum(singular_values_squared)

explained_energy = (
    singular_values_squared / total_energy
)

cumulative_energy = np.cumsum(explained_energy)

# print("Individual Explained Energy:")
# print(explained_energy)

# print("\nCumulative Explained Energy:")
# print(cumulative_energy)

# Step 7: Truncated SVD

K = 2

U_k = U[:, :K]

sigma_k = sigma[:K]

VT_k = Vt[:K, :]

# print("U_k Shape:", U_k.shape)
# print("Sigma_k Shape:", sigma_k.shape)
# print("VT_k Shape:", VT_k.shape)

# Step 8: Construct Sigma matrix

Sigma_k = np.diag(sigma_k)

# print("Sigma_k:")
# print(Sigma_k)

# Reconstruct the matrix

reconstructed_matrix = U_k @ Sigma_k @ VT_k

# print("\nReconstructed Matrix:")
# print(reconstructed_matrix)

# Step 9: Add User Means Back

predicted_ratings = (
    reconstructed_matrix + user_means.to_numpy()[:, np.newaxis]
)

print("Predicted Rating Matrix:")
print(predicted_ratings)

# Step 10: Compare Actual and Predicted Ratings

print("Actual Ratings:")
print(user_item_matrix)

print("\nPredicted Ratings:")
predicted_ratings_df = (pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    ))



def recommend_svd(top_k, user_id):

    if user_id not in predicted_ratings_df.index:
        raise ValueError(f"User {user_id} not found.")

    user_prediction = predicted_ratings_df.loc[user_id]

    rated_movies = user_item_matrix.loc[user_id].dropna().index

    unseen_movies = user_prediction.drop(rated_movies)

    recommedations = unseen_movies.sort_values(ascending = False)

    return recommedations.head(top_k)

print("User 1 Recommendations:")
print(recommend_svd(user_id=1, top_k=2))

print("\nUser 2 Recommendations:")
print(recommend_svd(user_id=2, top_k=2))

print("\nUser 3 Recommendations:")
print(recommend_svd(user_id=3, top_k=2))