import pandas as pd

class DataValidator:

    def __init__(self):
        self.issues = []

    def add_issue(self, dataset, rule, count, severity):
        if count > 0:
            self.issues.append({
                "dataset": dataset,
                "rule": rule,
                "count": int(count),
                "severity": severity
            })

    def validate_columns(self, name, df, expected):
        missing = set(expected) - set(df.columns)
        unexpected = set(df.columns) - set(expected)

        self.add_issue(
            name, "Missing columns", len(missing), "ERROR"
        )

        self.add_issue(
            name, "Unexpected columns", len(unexpected), "WARNING"
        )

    def validate_nulls(self, name, df, required_columns):
        for column in required_columns:
            self.add_issue(
                name,
                f"Missing values in {column}",
                df[column].isna().sum(),
                "ERROR"
            )

    def validate_primary_key(self, name, df, columns):
        if not columns:
            return

        duplicates = df.duplicated(subset=columns).sum()

        self.add_issue(
            name,
            f"Duplicate primary key: {columns}",
            duplicates,
            "ERROR"
        )

    def validate_positive_ids(self, name, df, columns):
        for column in columns:
            invalid = (df[column] <= 0).sum()

            self.add_issue(
                name,
                f"Non-positive {column}",
                invalid,
                "ERROR"
            )

    def validate_ratings(self, ratings):
        invalid_range = (
            ~ratings["rating"].between(0.5, 5.0)
        ).sum()

        invalid_steps = (
            (ratings["rating"] * 2) % 1 != 0
        ).sum()

        duplicate_pairs = ratings.duplicated(
            subset=["userId", "movieId"]
        ).sum()

        self.add_issue(
            "ratings", "Invalid rating range",
            invalid_range, "ERROR"
        )

        self.add_issue(
            "ratings", "Invalid rating increments",
            invalid_steps, "ERROR"
        )

        self.add_issue(
            "ratings", "Duplicate user-movie pairs",
            duplicate_pairs, "ERROR"
        )

    def validate_references(self, movies, ratings, tags, links):
        movie_ids = set(movies["movieId"])

        checks = [
            ("ratings", ratings),
            ("tags", tags),
            ("links", links)
        ]

        for name, df in checks:
            invalid = (~df["movieId"].isin(movie_ids)).sum()

            self.add_issue(
                name,
                "Unknown movie references",
                invalid,
                "ERROR"
            )

    def validate_metadata(self, movies, links):
        duplicate_titles = movies["title"].duplicated().sum()

        self.add_issue(
            "movies",
            "Repeated movie titles",
            duplicate_titles,
            "INFO"
        )

        self.add_issue(
            "links",
            "Missing TMDb identifiers",
            links["tmdbId"].isna().sum(),
            "WARNING"
        )

    def validate_timestamps(self, name, df):
        parsed = pd.to_datetime(
            df["timestamp"],
            unit="s",
            errors="coerce",
            utc=True
        )

        invalid = parsed.isna().sum()

        self.add_issue(
            name,
            "Invalid timestamps",
            invalid,
            "ERROR"
        )

    def run_all(self, datasets):
        self.issues = []

        movies = datasets["movies"]
        ratings = datasets["ratings"]
        tags = datasets["tags"]
        links = datasets["links"]

        self.validate_columns(
            "movies", movies,
            ["movieId", "title", "genres"]
        )

        self.validate_columns(
            "ratings", ratings,
            ["userId", "movieId", "rating", "timestamp"]
        )

        self.validate_columns(
            "tags", tags,
            ["userId", "movieId", "tag", "timestamp"]
        )

        self.validate_columns(
            "links", links,
            ["movieId", "imdbId", "tmdbId"]
        )

        self.validate_nulls(
            "movies", movies, ["movieId", "title", "genres"]
        )

        self.validate_nulls(
            "ratings", ratings,
            ["userId", "movieId", "rating", "timestamp"]
        )

        self.validate_nulls(
            "tags", tags,
            ["userId", "movieId", "tag", "timestamp"]
        )

        self.validate_nulls(
            "links", links, ["movieId", "imdbId"]
        )

        self.validate_primary_key(
            "movies", movies, ["movieId"]
        )

        self.validate_primary_key(
            "ratings", ratings, ["userId", "movieId"]
        )

        self.validate_primary_key(
            "links", links, ["movieId"]
        )

        self.validate_positive_ids(
            "movies", movies, ["movieId"]
        )

        self.validate_positive_ids(
            "ratings", ratings, ["userId", "movieId"]
        )

        self.validate_positive_ids(
            "tags", tags, ["userId", "movieId"]
        )

        self.validate_positive_ids(
            "links", links, ["movieId"]
        )

        self.validate_ratings(ratings)

        self.validate_references(
            movies, ratings, tags, links
        )

        self.validate_metadata(movies, links)

        self.validate_timestamps("ratings", ratings)
        self.validate_timestamps("tags", tags)

        return self.issues

