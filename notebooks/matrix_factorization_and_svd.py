
import pandas as pd
import numpy as np


rng = np.random.default_rng()

# P = np.array([[0.9,0.1],[0.7,0.2],[0.1,0.9]])
# Q = np.array([[0.9,0.1],[0.1,0.9],[0.2,0.8]])
# transpose_matrix = np.transpose(Q)
# # print(transpose_matrix)
# dot_prod = np.dot(P,transpose_matrix)
# print(dot_prod)

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

user_item_matrix = (ratings_df.pivot(index='user_id',columns='movie_title',values='rating'))
# print(user_item_matrix)

rows,cols = user_item_matrix.shape[0],user_item_matrix.shape[1]
missing_values = user_item_matrix.isna().to_numpy().sum()
#missing_values = user_item_matrix.isna().sum().sum()

possible_ratings = rows*cols

sparsity_percentage = (1-((possible_ratings-missing_values)/possible_ratings))*100

# sparsity_percentage


A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])
# Transpose of Matrix
transpose_A = A.T

# Element wise matrix multiplication of A & B.
# C = [[0,0],[0,0]]
# length = len(A)
# for i in range(length):
#     for j in range(length):
#         for k in range(length):
#             C[i][j] += A[i][k]*B[k][j]
# print(C)

numpy_matrix_multiplication = A @ B
# print(C)
# print(numpy_matrix_multiplication)

P = rng.integers(low=1,high=9,size=(4,2))
Q = rng.integers(low=1,high=9,size=(3,2))
transpose_Q = Q.T

R = np.array([[5,3,1],[4,2,1],[1,1,5]])

U,S,Vt = np.linalg.svd(R)

# print("Original Matrix:\n", R)
# print("\nU:\n", U)
# print("\nSingular Values:\n", S)
# print("\nV Transpose:\n", Vt)

Sigma = np.diag(S)

R_reconstructed = U @ Sigma @ Vt

# print("Reconstructed: \n",R_reconstructed)

# print(np.allclose(R,R_reconstructed))

#selecting the number of latent dimentions

def Construtive_Matrix(k,U,S,Vt):
    # Truncate Matrices 

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k,:]

    # Converting singular values into diagonal values

    Sigma_k = np.diag(np.sqrt(S_k))
    
    # Reconstruct approximate rating matrix
    # R_approx = U_k @ Sigma_k @ Vt_k


    P = U_k @ Sigma_k
    Q = Vt_k.T @ Sigma_k
    print("User factors:", P.shape)
    print("Item factors:", Q.shape)
    
    R_predicted = P @ Q.T

    print(R_predicted)

    predicted_rating = np.dot(P[0], Q[1])

    print(predicted_rating)

    # print("U_k shape:", U_k.shape)
    # print("Sigma_k shape:", Sigma_k.shape)
    # print("Vt_k shape:", Vt_k.shape)

    # print("\nApproximate matrix:\n", R_approx)

    # Need to Measure info Retained

    # total_energy = np.sum(S ** 2)

    # retained_energy = np.sum(S_k ** 2)

    # explained_ratio = retained_energy / total_energy

    # print("Information retained:", explained_ratio * 100, "%")

    # error = np.linalg.norm(R - R_approx, ord="fro")

    # print("Reconstruction error:", error)
# print("K = 1,\n")
# Construtive_Matrix(1,U,S,Vt)
# # print('..........................................')
# print("K = 2,\n")
Construtive_Matrix(2,U,S,Vt)
# print('..........................................')
# print("K = 3,\n")
# Construtive_Matrix(3,U,S,Vt)
print("User latent factors:", P.shape)
print("Item latent factors:", Q.shape)

print("\nUser factors:\n", P)
print("\nItem factors:\n", Q)