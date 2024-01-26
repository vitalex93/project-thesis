from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.embeddings import OpenAIEmbeddings
#import yake



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
    return ' '.join(filtered_sentence)

# def extract_keywords(text, stopwords=None, top=10, n=3, dedupLim=0.9):
#         kw_extractor = yake.KeywordExtractor(lan='en', top=top, n=n,
#                                              dedupLim=dedupLim, stopwords=stopwords)
#         keywords = kw_extractor.extract_keywords(text)
#         kw = [kw[0] for kw in keywords]
#         return ' '.join(kw)

def calculate_percentage(list1, list2, metric):
    count = 0
    for element in list1:
        if element in list2:
            count += 1
    if metric == 'precision':
        percentage = (count / len(list1)) * 100
    elif metric == 'recall':
        percentage = (count / len(list2)) * 100
    elif metric == 'r_precision':
        count = 0
        for element in list1[:len(list2)]:
            if element in list2:
                count += 1
        percentage = (count / len(list2)) * 100
    
    return percentage

def encode(text, model):
    tm = model
    encoder = tm.transform([text]).toarray()[0]
    return encoder

def encode_w2v(sentence, model):
    words = sentence.split()
    embeddings = []
    for word in words:
        if word in model:
            embeddings.append(model[word])
        elif word == 'instalments':
            embeddings.append(model['installments'])
    if len(embeddings) > 0:
        sentence_embedding = sum(embeddings) / len(embeddings)
        return sentence_embedding

