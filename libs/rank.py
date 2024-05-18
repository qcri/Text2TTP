from typing import Optional

import os
import pickle

from transformers import (AutoTokenizer,
                          AutoModel,
                          T5ForConditionalGeneration)

from libs.pygaggle.rerank.base import Reranker
from libs.pygaggle.rerank.bm25 import Bm25Reranker
from libs.pygaggle.rerank.transformer import (MonoT5, 
                                              DuoT5, 
                                              UnsupervisedTransformerReranker,
                                              SentenceTransformersReranker)
from libs.pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from libs.pygaggle.model import SimpleBatchTokenizer
from libs.pygaggle.rerank.base import Query, Text

from copy import deepcopy


# Models
def construct_bm25() -> Reranker:
    return Bm25Reranker()


def construct_monot5(model_path = 'castorini/monot5-base-msmarco-10k', batch_size=96, device='cuda') -> Reranker:
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = MonoT5.get_tokenizer(model_path, batch_size=batch_size)
    return MonoT5(model=model, tokenizer=tokenizer)


def construct_duot5(model_path = 'castorini/duot5-base-msmarco', batch_size=96, device='cuda') -> Reranker:
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = DuoT5.get_tokenizer(model_path, batch_size=batch_size)
    return DuoT5(model=model, tokenizer=tokenizer)


def construct_sentsecbert(model_path = 'models/SentSecBERT3', device='cuda') -> Reranker:
    return SentenceTransformersReranker(model_path, device=device)


def construct_attackbert(model_path = 'basel/ATTACK-BERT', device='cuda') -> Reranker:
    return SentenceTransformersReranker(model_path, device=device)


def load_cache_or_run(cache_file, runner, force=False, verbose=True, **kwargs):
    if os.path.exists(cache_file) and not force:
        if verbose:
            print('Loading from cache file:', cache_file)
            
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        if verbose:
            if force:
                print('Forced not to use cache')
            else:
                print('No cache found at:', cache_file, 'or forced!')
            
        run_exmp, run_metrics = runner(**kwargs)
        with open(cache_file, 'wb') as f:
            pickle.dump((run_exmp, run_metrics), f)
            
        if verbose:
            print('Saved to cache file:', cache_file)
            
        return run_exmp, run_metrics


# Preprocess
def get_texts(corpus, text_col:str = 'text', id_col:str = 'tech_id'):
    texts_indexed = [(idx, r[id_col], r[text_col]) for idx, (_, r) in enumerate(corpus.iterrows())]
    texts = [Text(sent, dict(docid=doc_id)) for _, doc_id, sent in texts_indexed]
    label_map = {doc_id:idx for idx, doc_id, _ in texts_indexed}

    return texts, label_map


def get_queries(queries, text_col:str = 'text', label_col: Optional[str] = None):
    if label_col is not None:
        return [Query(row[text_col], metadata=dict(label=row[label_col])) for _, row in queries.iterrows()]
    return [Query(row[text_col]) for _, row in queries.iterrows()]


def swap_text(examples, new_texts):
    new_examples = deepcopy(examples) 
    for example in new_examples:
        docs = example.documents
        for doc in docs:
            next_text = new_texts[doc.metadata['docid']]
            doc.text = next_text
            
    return new_examples


def print_error(errors, idx, doc_topn=5, doc_text=False):
    error_exmpl = [exmpl for exmpl in errors if not any(exmpl.labels)]
    print(f"{idx + 1} error out of {len(error_exmpl)}\n")

    print(f'Query: {error_exmpl[idx].query.text}')
    print(f'Label: {error_exmpl[idx].query.metadata["label"]}\n')

    for doc in error_exmpl[idx].documents[:doc_topn]:
        doc_out = f'Score: {doc.score}, Tech: {doc.metadata["docid"]}'
        if doc_text:
            doc_out += f', Text: {doc.text}\n'
        print(doc_out)