from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
words = [
    "astronaut",
    "Astronauts",
    "travel",
    "travels",
    "travelling",
    "survive",
    "survives",
    "survival",
    "surviving"
]
# print("************************************Stemmed Words************************************")
# for word in words:
#     print(word, "->", stemmer.stem(word))


# print("************************************Lemmatized Words************************************")
# for word in words:
#     print(word, "->", lemmatizer.lemmatize(word))

print(lemmatizer.lemmatize("running", pos="v"))
print(lemmatizer.lemmatize("travelling", pos="v"))
print(lemmatizer.lemmatize("surviving", pos="v"))