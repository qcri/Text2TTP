import os
import re

import pandas as pd
from newspaper.nlp import split_sentences
from ioc_fanger import fang

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 


def load_source_sent(source_hash, data_dir='data/mitre'):
    """
    Load sentences from reports

    :param source_hash: Hash (source url) of the report.
    :param data_dir: Directory of scraped and parsed (newspaper) data.
    :return: `list` of sentences
    """
    with open(os.path.join(data_dir, 'content', source_hash), 'r', encoding='utf-8') as f:
        content = f.read().replace('\n\n', ' ')
        content = fang(content)

    sents = split_sentences(content)
    return sents


def _filter_sanitized(df, sanitize=True):
    if sanitize:
        return df.drop(columns=['text']).rename(columns={'clean': 'text'})
    else:
        return df.drop(columns=['clean'])


def load_mitre_kb(path='data/mitre/aggregated.sanitized.csv', sanitize=True, sep_cleaned=False, n=-1):
    """
    Load MITRE KB

    :param path: Path to the MITRE KB dataset.
    :param sanitize: Retrive IoC sanitized sentences.
    :param sep_cleaned: Retrive both sanitized and raw sentences (can't be set alongside `sanitize`).
    :return: `pd.DataFrame` containing MITRE KB
    """
    assert sep_cleaned == False or sanitize == False, "Both `sanitize` and `sep_cleaned` can't be true."
    raw_dataset = pd.read_csv(path)
    if not sep_cleaned:
        raw_dataset = raw_dataset.drop(columns=['sanitized'])
        raw_dataset = _filter_sanitized(raw_dataset, sanitize)
       
    if n > 0:
        tech_counts = raw_dataset.value_counts("tech_id").to_frame(name="count").reset_index()
        topn = tech_counts.loc[:n-1, 'tech_id'].values
        dataset = pd.DataFrame(raw_dataset[raw_dataset['tech_id'].isin(topn)]).reset_index(drop=True)
    else:
        dataset = raw_dataset.reset_index(drop=True)
    return pd.DataFrame(dataset)


def load_annotated(path='data/sentences.sanitized.csv', sanitize=True, sep_cleaned=False):
    """
    Load annotated report sentences

    :param path: Path to the report sentence dataset.
    :param sanitize: Retrive IoC sanitized sentences.
    :param sep_cleaned: Retrive both sanitized and raw sentences (can't be set alongside `sanitize`).
    :return: `pd.DataFrame` annotated report sentences
    """
    assert sep_cleaned == False or sanitize == False, "Both `sanitize` and `sep_cleaned` can't be true."

    man_sent = pd.read_csv(path)   
    if not sep_cleaned:
        man_sent = man_sent.drop(columns=['sanitized'])
        man_sent = _filter_sanitized(man_sent, sanitize)

    return man_sent


def cleanup(s):
    """
    Clean up MITRE sentences and descriptions
    
    :param s: Sentence or a Description
    :return: Cleanend `str`
    """
    
    s = ' '.join(s.split()) # remove aditional whitespaces
    s = re.sub(r'\[(\s*[0-9]*\s*)\]|\s\s+', '', s).strip() # remove references
    return s


def cleanup_text(text):
    """
    Clean up sentences and descriptions (Improved version)
    
    :param text: Sentence or a Description
    :return: Cleanend `str`
    """
    text = text.lower()

    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r"http\S+", "",text) #Removing URLs 

    html = re.compile(r'<.*?>') 
    text = html.sub(r'',text) #Removing html tags

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #Removing punctuations

    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text) #removing stopwords

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) #Removing emojis
    return text


def cleanup_texts(texts):
    """
    Clean up sentences and descriptions (Improved version)
    
    :param text: Sentence or a Description
    :return: Cleanend `str`
    """
    out = []
    for text in texts:
        out.append(cleanup_text(text))
    return out


def sent_tokenize(paragraphs):
    """
    Segment texts into sentences
    
    :param paragraphs: List of texts
    :return: Nested list of texts segmented into sentences
    """
    import stanza

    nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)

    tokenized = [[sent.text for sent in nlp(para).sentences] for para in paragraphs]
    return tokenized
