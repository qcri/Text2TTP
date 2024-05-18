import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from libs.sbert_transformer import SBertTransformer


def get_bert(model="bert-base-uncased", output_hidden_states=False, use_fast_tokenizer=True):
    """
    Get an instance of Bert model
    
    :param model: Name or path of the Bert model
    :param output_hidden_states: Whether to output hidden states or not
    :param use_fast_tokenizer: Wheather to use the fast tokenizer implementation
    :return: Tuple of `BertModel` and `BertTokenizer`
    """
    bert_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast_tokenizer)
    bert_model = AutoModel.from_pretrained(model, output_hidden_states=output_hidden_states).eval()
    return bert_model, bert_tokenizer


def get_cybert(model='../../models/CyBERT', output_hidden_states=False):
    """
    Get an instance of CyBert model
    
    :param model: Path to the CyBert model
    :param output_hidden_states: Whether to output hidden states or not
    :return: Tuple of `BertModel` and `BertTokenizer`
    """
    return get_bert(model, output_hidden_states, False)


def get_secbert(model="jackaduma/SecBERT", output_hidden_states=False):
    """
    Get an instance of SecBert model
    
    :param output_hidden_states: Whether to output hidden states or not
    :return: Tuple of `BertModel` and `BertTokenizer`
    """
    return get_bert(model, output_hidden_states)


def get_sbert(model='all-MiniLM-L6-v2'):
    """
    Get an instance of Sentence Bert model
    
    :param model: Name of the pre-trained model
    :return: `SentenceTransformer` model
    """
    return SentenceTransformer(model)


def get_sent_cybert(path):
    """
    Get an instance of CyBERT model
    
    :param model: Path of the pre-trained model
    :return: `SentenceTransformer` model
    """
    from sentence_transformers.models import Pooling
    
    word_embedding_model = SBertTransformer(path)    

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def cos_sim(a, b):
    """
    Cosine similary between cross entries of `a` and `b`
    
    :param a: `list` of strings
    :param b: `list` of stirngs
    :return: `ndarray` of cross-joined cosine similarity scores
    """
    return util.cos_sim(a, b)


def sample_pos_neg(df, frac=1.0):
    """
    Generate positive/negative pairs from sentences
    
    :param group: `pd.DataFrame` of strings with columns `Sentence` and `Technique`
    """
    def cross_join(group, drop_duplicate='sentence'):
        group['key'] = 0
        cross = group.merge(group, on='key', how='outer')
        
        if drop_duplicate == 'sentence':
            cross = cross[~(cross.Sentence_x == cross.Sentence_y)]
        elif drop_duplicate == 'technique':
            cross = cross[~(cross.Technique_x == cross.Technique_y)]
        else:
            raise NotImplementedError("Not supported!")
            
        cross = cross.loc[:, ['Sentence_x', 'Sentence_y']]
        cross.columns = ['a', 'b']
        return cross
    
    all_pos = pd.concat([cross_join(group) for tech, group in df.groupby('Technique')]).sample(frac=frac)
    all_pos['label'] = 1.0
    
    all_neg = cross_join(df, drop_duplicate='technique')
    all_neg['label'] = 0.0
    
    sampled_neg = all_neg.sample(all_pos.shape[0])

    return pd.concat([all_pos, sampled_neg]).reset_index(drop=True)


# Predictions
def get_hidden(model, tokenizer, texts, max_length=None):
    """
    Get hidden states for given text input
    
    :param model: Bert model
    :param tokenizer: Bert tokenizer
    :param texts: List of text as input to the model
    :return: Hidden states of the Bert model
    """
    hidden_states = []
    with torch.no_grad():
        for text in texts:
            if max_length is not None:
                encoded_input = tokenizer(text, return_tensors='pt', max_length=max_length)
            else:
                encoded_input = tokenizer(text, return_tensors='pt')
            outputs = model(**encoded_input)

            hidden_states.append(outputs[2])
            
    return hidden_states


def token_encode(model, tokenizer, texts):
    """
    Get embeddings for each token in texts (hidden state of the second to last Bert layer)
    
    :param model: Bert model
    :param tokenizer: Bert tokenizer
    :param texts: List of text as input to the model
    :return: Embeddings of the tokens in a nested list
    """
    texts_token_embs = []
    for h in get_hidden(model, tokenizer, texts):
        token_hidden = torch.stack(h, dim=0).squeeze(dim=1).permute(1,0,2)
        token_embs = [t[-2] for t in token_hidden]
        texts_token_embs.append(token_embs)
    return texts_token_embs


def sent_encode(model, tokenizer, texts, max_length=None):
    """
    Get embeddings for given text input (hidden state of the second to last Bert layer)
    
    :param model: Bert model
    :param tokenizer: Bert tokenizer
    :param texts: List of text as input to the model
    :return: Embeddings of the input texts
    """
    texts_sent_embs = []
    for h in get_hidden(model, tokenizer, texts, max_length):
        sent_hidden = h[-2][0]
        sent_emb = torch.mean(sent_hidden, dim=0)
        texts_sent_embs.append(sent_emb)
    return torch.stack(texts_sent_embs)