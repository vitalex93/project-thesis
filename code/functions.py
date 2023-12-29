import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Define the function to extract named entities
def extract_named_entities(text_descriptions, nlp):
    named_entities_list = []

    for text in text_descriptions:
        doc = nlp(text)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        named_entities_list.append(named_entities)

    return named_entities_list

def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    #with no lower case conversion
    filtered_sentence = []
    
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

def get_top_k_synonyms(word, model="word2vec-google-news-300",k=5):
    # Load pre-trained Word2Vec embeddings (you can use other embeddings as well)
    word_vectors = api.load(model)

    # Check if the word is in the vocabulary
    if word not in word_vectors:
        return []

    # Get the vector representation of the input word
    input_word_vector = word_vectors[word]

    # Calculate cosine similarity between the input word and all words in the vocabulary
    cosine_similarities = cosine_similarity([input_word_vector], word_vectors.vectors)

    # Get the indices of the top k most similar words (excluding the input word itself)
    most_similar_indices = np.argsort(cosine_similarities[0])[::-1][1:k+1]

    # Get the actual words for the most similar indices
    synonyms = [word_vectors.index_to_key [idx] for idx in most_similar_indices]

    return synonyms


