import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import string


# Ensure required resources are downloaded
nltk.download('stopwords')

tokenizer = TreebankWordTokenizer()  # for tokenization
stemmer = PorterStemmer()  # 
stopwords = nltk.corpus.stopwords.words('english') # for stopwords removal
lemmatizer = WordNetLemmatizer() # for lemmatization


def tokenize(sentence: str) -> list:
    """
        split sentence into array of words/tokens
        a token can be a word or punctuation character, or number
    """
    return tokenizer.tokenize(sentence)


def remove_punctuation(tokens):
    """
    remove some punctuations symbols
    """
    return [token for token in tokens if token not in string.punctuation]

def spelling_correction(tokens):
    """
    Correcting spelling mistakes
    """
    return [str(TextBlob(token).correct()) for token in tokens]


def case_folding(tokens: list) -> list:
    """
    lower casing all tokens
    """
    return [token.lower() for token in tokens]


def remove_stopword(tokens: list) -> list:
    """
    removing words which occur frequently but carry little information
    """
    return [token for token in tokens if token not in stopwords]


def stemming(tokens: list) -> list:
    """
    removing word suffix and prefix
    """
    return [stemmer.stem(token) for token in tokens]


def lemmanization(tokens: list) -> list:
    """
    Reduce word to its lemma form
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [word for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

