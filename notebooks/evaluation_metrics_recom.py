
def precision_at_k(actual, recommended, k):
    if k <= 0:
        raise ValueError("K must be greater than 0")

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in actual)

    return round(hits / k, 3)


def recall_at_k(actual, recommended, k):
    if k <= 0:
        raise ValueError("K must be greater than 0")

    if len(actual) == 0:
        return None

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in actual)

    return round(hits / len(actual), 3)


def f1_at_k(actual, recommended, k):
    precision = precision_at_k(actual, recommended, k)
    recall = recall_at_k(actual, recommended, k)

    if recall is None:
        return None

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return round(
        (2 * precision * recall) / denominator, 3
    )


def average_precision(actual, recommended):
    if len(actual) == 0:
        return None

    relevant_hits = 0
    precision_sum = 0.0

    for position, item in enumerate(recommended, start=1):
        if item in actual:
            relevant_hits += 1
            precision_sum += relevant_hits / position

    return round(precision_sum / len(actual), 3)


def mean_average_precision(ap_values):
    valid_scores = [
        score for score in ap_values
        if score is not None
    ]

    if not valid_scores:
        return None

    return round(sum(valid_scores) / len(valid_scores), 3)


def evaluate_recommendations(actual, recommended, k):
    return {
        "precision": precision_at_k(actual, recommended, k),
        "recall": recall_at_k(actual, recommended, k),
        "f1": f1_at_k(actual, recommended, k),
        "ap": average_precision(actual, recommended)
    }


def evaluate_dataset(ground_truth, recommendations, k):

    if not ground_truth:
        raise ValueError("Ground truth dataset is empty.")
    
    missing = set(ground_truth) - set(recommendations)

    if missing:
        raise ValueError(
            f"Missing recommendations for: {sorted(missing)}"
        )

    all_results = {}
    ap_values = []

    for movie, actual in ground_truth.items():
        recommended = recommendations[movie]

        result = evaluate_recommendations(
            actual, recommended, k
        )

        all_results[movie] = result
        ap_values.append(result["ap"])

    map_score = mean_average_precision(ap_values)

    return all_results, map_score



ground_truth = {
    "Interstellar": {"The Martian", "Gravity"},
    "The Martian": {"Interstellar", "Gravity"},
    "Gravity": {"Interstellar", "The Martian"},
    "Titanic": {"The Notebook"},
    "The Notebook": {"Titanic"}
}

recommendations = {
    "Interstellar": ["The Martian", "Gravity", "The Notebook"],
    "The Martian": ["Gravity", "Interstellar", "Titanic"],
    "Gravity": ["The Martian", "Interstellar", "The Notebook"],
    "Titanic": ["The Notebook", "Finding Nemo", "The Lion King"],
    "The Notebook": ["Titanic", "The Lion King", "Avengers"]
}

K = 5

all_results, map_score = evaluate_dataset(
    ground_truth, recommendations, K
)

for movie, metrics in all_results.items():
    print(f"\nMovie: {movie}")
    print(f"Precision@{K}: {metrics['precision']}")
    print(f"Recall@{K}:    {metrics['recall']}")
    print(f"F1@{K}:        {metrics['f1']}")
    print(f"AP:            {metrics['ap']}")

print(f"\nMAP@{K}: {map_score}")
