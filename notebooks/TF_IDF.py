import math

documents = [
    "cat likes milk",
    "cat likes fish",
    "dog likes fish"
]

tokenized_documents = [doc.split() for doc in documents]

vocabulary = sorted(
    set(word for doc in tokenized_documents for word in doc)
)

def calculate_tf(document, vocabulary):
    total_words = len(document)

    tf = {}

    for word in vocabulary:
        count = document.count(word)
        tf[word] = count / total_words

    return tf


def calculate_df(tokenized_documents, vocabulary):

    df = {}

    for word in vocabulary:
        count = 0

        for document in tokenized_documents:
            if word in document:
                count += 1

        df[word] = count

    return df

def calculate_idf(df, total_documents):

    idf = {}

    for word, document_frequency in df.items():
        idf[word] = math.log(
            total_documents / document_frequency
        )

    return idf

df = calculate_df(tokenized_documents, vocabulary)

idf = calculate_idf(
    df,
    len(tokenized_documents)
)

# print(idf)

def calculate_tfidf(document, vocabulary, idf):

    tf = calculate_tf(document, vocabulary)

    tfidf = {}

    for word in vocabulary:
        tfidf[word] = tf[word] * idf[word]

    return tfidf

for document in tokenized_documents:

    tfidf = calculate_tfidf(
        document,
        vocabulary,
        idf
    )

    # print(tfidf)

def create_vector(tfidf, vocabulary):

    return [
        tfidf[word]
        for word in vocabulary
    ]

tfidf_vectors = []

for document in tokenized_documents:

    tfidf = calculate_tfidf(
        document,
        vocabulary,
        idf
    )

    vector = create_vector(
        tfidf,
        vocabulary
    )

    tfidf_vectors.append(vector)

# print(tfidf_vectors)

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)

similarity_12 = cosine_similarity(
    tfidf_vectors[0],
    tfidf_vectors[1]
)

similarity_13 = cosine_similarity(
    tfidf_vectors[0],
    tfidf_vectors[2]
)

similarity_23 = cosine_similarity(
    tfidf_vectors[1],
    tfidf_vectors[2]
)

# print("D1-D2:", similarity_12)
# print("D1-D3:", similarity_13)
# print("D2-D3:", similarity_23)

# from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "cat likes milk",
    "cat likes fish",
    "dog likes fish"
]

# vectorizer = TfidfVectorizer()

# tfidf_matrix = vectorizer.fit_transform(documents)

# print("Vocabulary:")
# print(vectorizer.vocabulary_)

# print("\nTF-IDF Matrix:")
# print(tfidf_matrix.toarray())

# print("\nFeature Names:")
# print(vectorizer.get_feature_names_out())

# from sklearn.metrics.pairwise import cosine_similarity

# similarity_matrix = cosine_similarity(tfidf_matrix)

# print(similarity_matrix)

# similarity_matrix = cosine_similarity(tfidf_matrix)

# print(similarity_matrix)