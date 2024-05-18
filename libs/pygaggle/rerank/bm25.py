from copy import deepcopy
import math
import re

import numpy as np

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from .base import Reranker, Query, Text


__all__ = ['Bm25Reranker']


class Bm25Reranker(Reranker):
    def __init__(self,
                 k1: float = 1.6,
                 b: float = 0.75,
                 tokenize_method: str = 'nlp'):
        self.k1 = k1
        self.b = b
        self.tokenize_method = tokenize_method

    def rescore(self, query: Query, texts: list[Text]) -> list[Text]:
        query_words = self.tokenize(query.text)
        sentences = self.tokenize([t.text for t in texts])
        
        bm25 = BM25Okapi(sentences, k1=self.k1, b=self.b)
        scores = bm25.get_scores(query_words)
        
        texts = deepcopy(texts)
        for score, text in zip(scores, texts):
            text.score = score
        return texts

    def tokenize(self, texts):
        is_single = False
        if isinstance(texts, str):
            is_single = True
            texts = [texts]
            
        if self.tokenize_method == 'nlp':
            stop_blank = set(stopwords.words('english')).union(set([' ']))

            texts = [word_tokenize(t) for t in texts]
            texts = [[re.sub("[^0-9a-zA-Z]", " ", token) for token in t] for t in texts]
            texts = [[token for token in t if token not in stop_blank] for t in texts]
        else:
            raise NotImplemented("Tokenize method not implemented")
        return texts[0] if is_single else texts
