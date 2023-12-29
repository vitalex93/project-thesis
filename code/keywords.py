import yake
from rake_nltk import Rake
#from gensim.summarization import keywords
#from gensim.summarization import textrank

class KeywordExtractor:
    def __init__(self, language='en', max_keywords=10):
        self.language = language
        #self.max_keywords = max_keywords
    
    def __call__(self, mode: str, text, stopwords=None, top=10, n=3, dedupLim=0.9,
                 max_kw=10, max_length=3, min_length=2, ranking_metric='word_frequency'):
        if mode == 'rake':
            kw = self.extract_rake(text=text, max_kw=max_kw, stopwords=stopwords,
                                   max_length=max_length, min_length=min_length, ranking_metric=ranking_metric)
            kws = ' '.join(kw)
            return kw
        elif mode == 'yake':
            kw = self.extract_yake(text=text, stopwords=stopwords, top=top,
                                   n=n, dedupLim=dedupLim)
            kws = ' '.join(kw)
            return kw

    def extract_yake(self, text, stopwords=None, top=10, n=3, dedupLim=0.9):
        kw_extractor = yake.KeywordExtractor(lan=self.language, top=top, n=n,
                                             dedupLim=dedupLim, stopwords=stopwords)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    def extract_rake(self, text, max_kw=10, stopwords=None, max_length=3, min_length=2, ranking_metric='word_frequency'):
        r = Rake(stopwords=stopwords, max_length=max_length, min_length=min_length, ranking_metric=ranking_metric )
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()[:max_kw]