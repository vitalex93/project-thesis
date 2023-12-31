import spacy
import faiss
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import numpy as np
import pandas as pd
from lib import utils
import joblib
from functools import partial
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

class ReportGenerator:

    def __init__(self, description, model) -> None:
        self.description = description
        self.model = model
        self.bow = joblib.load('./models/models/bow_model.joblib')
        self.bow_bigrams = joblib.load('./models/models/bigrams_bow_model.joblib')
        self.bow_trigrams = joblib.load('./models/models/trigrams_bow_model.joblib')
        self.tfidf = joblib.load('./models/models/tfidf_model.joblib')
        self.tfidf_bigrams = joblib.load('./models/models/bigrams_tfidf_model.joblib')
        self.tfidf_trigrams = joblib.load('./models/models/trigrams_tfidf_model.joblib')
        self.annotator = self.load_annotator()
        self.mt_money = pd.read_csv("./vector-indexes/mt_money.csv", sep=';')
        self.mt_dates = pd.read_csv("./vector-indexes/mt_date.csv", sep=';')
        self.mt_lov = pd.read_csv("./vector-indexes/mt_lov.csv", sep=';')
        self.mt_items = pd.read_csv("./vector-indexes/mt_items.csv", sep=';')
        self.mt_money_index = self.mt_money.Description
        self.mt_items_index = self.mt_items.Description
        self.mt_dates_index = self.mt_dates.Description
        self.mt_lov_index = self.mt_lov.Description
        self.llm = OpenAI(temperature=0.2, openai_api_key = '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')
        self.sbert = SentenceTransformer("all-mpnet-base-v2")
        self.w2v_models = ['word2vec-google-news-300', 'glove-wiki-gigaword-50', 'fasttext-wiki-news-subwords-300', 'glove-wiki-gigaword-300']
        self.bow_models = ['bow', 'tfidf', 'bow_bigrams', 'tfidf_bigrams', 'bow_trigrams', 'tfidf_trigrams']
        self.encoder = self.get_encoder(model=self.model) #SentenceTransformer("all-mpnet-base-v2")
        self.data_dictionary_df = pd.read_csv('./reports/data-dictionary.csv', sep=';')
        self.rows_dca_apptype = pd.read_csv('./reports/rows_dca_apptype.csv', sep=';')
        self.rows_dca = pd.read_csv('./reports/rows_dca.csv', sep=';')
        self.rows_dca_assetclass = pd.read_csv('./reports/rows_dca_assetclass.csv')
        self.ground_truth = pd.read_csv('./reports/ground-truth.csv', sep=';')
        self.columns = self.get_columns()

  
    def process_description(self):
        stopwords_removed = utils.remove_stopwords(self.description)
        #keywords = utils.extract_keywords(stopwords_removed)
        return stopwords_removed

    def load_annotator(self):
        annotator = spacy.load('./ner-results/model-best')
        return annotator

    def get_entitites(self):
        preprocessed_description = self.process_description()
        ner_model = self.annotator
        doc = ner_model(preprocessed_description)
        labels = []
        values = []
        for ent in doc.ents:
            labels.append(ent.label_)
            values.append(ent.text)
        named_entities = pd.DataFrame.from_dict({'Type': labels, 'Value': values})
        return named_entities
    
    def merge_entities_data_dict(self):
        named_entities = self.get_entitites()
        merged_df = pd.merge(named_entities, self.data_dictionary_df, on='Type')
        return merged_df
    
    def visualize_entities(self):
        ner_model = self.load_annotator()
        doc = ner_model(self.description)
        spacy.displacy.render(doc, style="ent", jupyter=True)
    
    def get_encoder(self, model):
        if model == 'bow':
            tm = self.bow
            encoder = partial(utils.encode, model=tm)
            return encoder
        elif model == 'tfidf':
            tm = self.tfidf
            encoder = partial(utils.encode, model=tm)
            return encoder
        elif model == 'bow_bigrams':
            tm = self.bow_bigrams
            encoder = partial(utils.encode, model=tm)
            return encoder
        elif model == 'tfidf_bigrams':
            tm = self.tfidf_bigrams
            encoder = partial(utils.encode, model=tm)
            return encoder
        elif model == 'bow_trigrams':
            tm = self.bow_trigrams
            encoder = partial(utils.encode, model=tm)
            return encoder
        elif model == 'tfidf_trigrams':
            tm = self.tfidf_trigrams
            encoder = partial(utils.encode, model=tm)
            return encoder
        elif model in self.w2v_models:
            tm = api.load(model)
            encoder = partial(utils.encode_w2v, model=tm)
            return encoder
        elif model == 'sbert':
            encoder = self.sbert.encode
            return encoder 
            
    def build_index(self, idx, model):
        model = self.model
        encoder = self.encoder
        if model == 'sbert':
            dim = encoder(idx).shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(encoder(idx))
            return index
        elif model in self.bow_models + self.w2v_models:
            idxvalues = []
            for i in list(idx):
                idxvalues.append(encoder(i))
            vectors = np.array(idxvalues)
            dim = vectors.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
            return index

    def get_encoded_description(self):
        encoder = self.encoder
        vectors = encoder(self.description)
        return vectors
    
    def get_similar_vectors_in_index(self, idx, df):
        index = self.build_index(idx=idx)
        vector = self.get_encoded_description()
        svector = np.array(vector).reshape(1,-1)
        distances, I = index.search(svector, k=5)
        return df.loc[I[0]]
    
    def get_encoded_values_per_type(self):
        unique_values_per_type = self.get_unique_values_per_type()
        encoded_values = {}
        encoder = self.encoder
        for k in unique_values_per_type.keys():
            encoded_values[k] = list(map(encoder, unique_values_per_type[k]))
        return encoded_values
            
    def get_similar_vectors_in_index_per_value(self, type, k=5):
        types = self.get_unique_types()
        if type == 'MT_MONEY' and type in types:
            encoded_values = [x for x in self.get_encoded_values_per_type()['MT_MONEY'] if x is not None] 
            index = self.build_index(idx=self.mt_money_index, model=self.model)
            df = self.mt_money
            list_of_value_names = self.get_unique_values_per_type()['MT_MONEY']
            top_k_similar = {}
            for i in range(len(encoded_values)):
                svector = np.array(encoded_values[i]).reshape(1,-1)
                distances, I = index.search(svector, k=k)
                top_k_similar[list_of_value_names[i]] = list(df.loc[I[0]]['Name'])
            return top_k_similar               
        if type == 'MT_ITEMS' and type in types:
            encoded_values = [x for x in self.get_encoded_values_per_type()['MT_ITEMS'] if x is not None] 
            index = self.build_index(idx=self.mt_items_index, model=self.model)
            df = self.mt_items
            list_of_value_names = self.get_unique_values_per_type()['MT_ITEMS']
            top_k_similar = {}
            for i in range(len(encoded_values)):
                svector = np.array(encoded_values[i]).reshape(1,-1)
                distances, I = index.search(svector, k=k)
                top_k_similar[list_of_value_names[i]] = list(df.loc[I[0]]['Name'])
            return top_k_similar 
        else:
            return {}
    
    def get_unique_types(self):
        merged_df = self.merge_entities_data_dict()
        return list(merged_df.Type.unique())
    
    def get_unique_values_per_type(self):
        merged_df = self.merge_entities_data_dict()
        unique_types = self.get_unique_types()
        unique_values_dict = {}
        for t in unique_types:
            filtered = merged_df[merged_df['Type'] == t]
            unique_values_dict[t] = list(filtered.Value.unique())
        return unique_values_dict

    def get_columns(self):
        mt_money = self.get_similar_vectors_in_index_per_value(type='MT_MONEY', k=1)
        mt_items = self.get_similar_vectors_in_index_per_value(type='MT_ITEMS', k=1)
        columns = []
        for k in mt_money.keys():
            if mt_money != {}:
                columns.append(mt_money[k][0])
        for k in mt_items.keys():
            if mt_items != {}:
                columns.append(mt_items[k][0])
        return columns
    
    def get_column_agg_function(self):
        measures = {**self.get_similar_vectors_in_index_per_value(type='MT_MONEY', k=1), **self.get_similar_vectors_in_index_per_value(type='MT_ITEMS', k=1)}
        template = PromptTemplate(
            input_variables=['measure', 'description'],
            template= 'Are there one or more aggregation functions, such as average, sum, median etc, that should be applied to this quantity \
                      {measure} according to this description?: {description}. Provide only the name of the function(s).'
                      )
        chain = LLMChain(llm=self.llm, prompt=template, verbose=False)
        agg_functions_dict = {}
        for measure, col in measures.items():
            agg_functions = chain.run({'measure':measure, 'description':self.description})
            agg_functions_dict[col[0]] = agg_functions 
        return agg_functions_dict

    def get_operator_type_for_lov(self, lov):
        template = PromptTemplate(
            input_variables=['lov', 'description'],
            template= 'Does the word/phrase {lov} act as a grouping or a filtering predicate in the following description?:{description}.\
                        If it is a grouping predicate return 1. If it is a filtering predicate return 0.')
        chain = LLMChain(llm=self.llm, prompt=template, verbose=False)
        operator = int(chain.run({'lov':lov, 'description':self.description}))
        return operator

    def get_rows(self):
        mt_lov = self.get_unique_values_per_type()['MT_LOV']
        mt_dates = self.get_unique_values_per_type()['MT_DATE']
        grouping_operators = []
        filtering_operators = []
        for lov in mt_lov + mt_dates:
            if self.get_operator_type_for_lov(lov=lov) == 1:
                grouping_operators.append(lov)
            elif self.get_operator_type_for_lov(lov=lov) == 0:
                filtering_operators.append(lov)
        if 'DCA' in grouping_operators and 'application type .' in grouping_operators and 'asset class' in grouping_operators:
            if 'DCA' in mt_lov and 'application type .' not in mt_lov and 'asset class' not in mt_lov:
                return self.rows_dca
            elif 'DCA' in mt_lov and 'asset class' in mt_lov and 'application type .' not in mt_lov:
                return self.rows_dca_assetclass
            elif 'DCA' in mt_lov and 'application type .' in mt_lov and 'asset class' not in mt_lov:
                return self.rows_dca_apptype
        else:
            return 'Not seen in given data'
        
    def evaluation(self, report, metric):
        columns = self.columns
        ground_truth = [x for x in list(self.ground_truth[report]) if not(pd.isnull(x)) == True]
        percentage = utils.calculate_percentage(columns, ground_truth, metric)
        return {report: percentage}







    

    

    




    