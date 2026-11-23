import pandas as pd
from pathlib import Path

path = Path("data/raw")

movies = pd.read_csv(path / 'movies.csv')
links = pd.read_csv(path / 'links.csv')
ratings = pd.read_csv(path / 'ratings.csv')
tags = pd.read_csv(path / 'tags.csv')

# 1. Primary key checks
print("\n--- PRIMARY KEY CHECKS ---")

print("Duplicate movie IDs:", movies["movieId"].duplicated().sum())
print("Duplicate link movie IDs:", links["movieId"].duplicated().sum())

print("Unique users:", ratings["userId"].nunique())
print("Unique movies:", movies["movieId"].nunique())


# 2. Duplicate user-movie rating pairs
print("\n--- RATING PAIR CHECKS ---")

duplicate_pairs = ratings.duplicated(
    subset=["userId", "movieId"]
).sum()

print("Duplicate user-movie pairs:", duplicate_pairs)


# 3. Rating validation
print("\n--- RATING VALIDATION ---")

valid_rating_range = ratings["rating"].between(0.5, 5.0)

valid_rating_steps = (
    (ratings["rating"] * 2) % 1 == 0
)

print("Invalid rating range:", (~valid_rating_range).sum())
print("Invalid rating increments:", (~valid_rating_steps).sum())


# 4. ID validation
print("\n--- ID VALIDATION ---")

for column in ["userId", "movieId"]:
    print(f"Ratings {column} <= 0:", (ratings[column] <= 0).sum())

print("Movies movieId <= 0:", (movies["movieId"] <= 0).sum())
print("Tags userId <= 0:", (tags["userId"] <= 0).sum())
print("Tags movieId <= 0:", (tags["movieId"] <= 0).sum())


# 5. Referential integrity
print("\n--- REFERENTIAL INTEGRITY ---")

movie_ids = set(movies["movieId"])

invalid_rating_movies = ~ratings["movieId"].isin(movie_ids)
invalid_tag_movies = ~tags["movieId"].isin(movie_ids)
invalid_link_movies = ~links["movieId"].isin(movie_ids)

print("Ratings with unknown movie IDs:", invalid_rating_movies.sum())
print("Tags with unknown movie IDs:", invalid_tag_movies.sum())
print("Links with unknown movie IDs:", invalid_link_movies.sum())


# 6. Movie metadata checks
print("\n--- MOVIE METADATA ---")

print("Duplicate titles:", movies["title"].duplicated().sum())
print("Empty titles:", movies["title"].str.strip().eq("").sum())
print("Empty genres:", movies["genres"].str.strip().eq("").sum())

print("\nRepeated movie titles:")
print(
    movies[movies["title"].duplicated(keep=False)]
    .sort_values("title")
)


# 7. Timestamp validation
print("\n--- TIMESTAMP VALIDATION ---")

for name, df in [
    ("ratings", ratings),
    ("tags", tags),
]:
    parsed = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
        utc=True
    )

    print(f"{name} invalid timestamps:", parsed.isna().sum())


# 8. External ID checks
print("\n--- EXTERNAL ID CHECKS ---")

print("Missing IMDb IDs:", links["imdbId"].isna().sum())
print("Missing TMDb IDs:", links["tmdbId"].isna().sum())