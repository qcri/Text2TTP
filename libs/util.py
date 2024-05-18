import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder


def plot_history(history, title=None):
    plt.figure(figsize=(10.5, 7))
    history_df = pd.DataFrame(history, columns=['Loss']).reset_index().rename(columns={'index': 'Epoch'})
    sns.lineplot(x='Epoch', y='Loss', data=history_df, linewidth=5).set(title="" if title is None else title)
    plt.show()
    
    
def plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, names):
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=names, yticklabels=names, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()
    

def label_encode(labels):
    label_map = {t: i for i, t in enumerate(labels.drop_duplicates().values)}
    return [label_map[t] for t in labels]


def score(labels, pred, proba=None, multi_class=True):
    accuracy = accuracy_score(labels, pred)
    
    avg = 'weighted' if multi_class else 'binary'
    f1 = f1_score(labels, pred, zero_division=0, average=avg)
    prec = precision_score(labels, pred,  zero_division=0, average=avg)
    recall = recall_score(labels, pred,  zero_division=0, average=avg)
    
    scores = {'acc': accuracy, 'f1': f1, 'prec': prec, 'recall': recall}
    
    if proba is not None:
        scores['auc'] = roc_auc_score(labels, proba, average='weighted', multi_class='ovr')
    
    return scores


def split_mean_score(scores):
    scores = pd.DataFrame(scores).reset_index().rename(columns={'index': 'Split'})
    scores.loc[len(scores)] = ['Mean', *pd.DataFrame(scores).drop(columns=['Split']).mean().to_list()]
    return scores
    

def plot_sent_vs_sent(sent, similarity):
    """
    Plot heatmap of similarity between sentences and sentences
    
    :param sent: `pd.DataFrame` containing sentences (must include `Sentence` and `Technique` columns)
    :param similarity: Similarity matrix
    """
    sent_tech_map = {row.Sentence: row.Technique for i, row in sent.iterrows()}

    sim_df = pd.DataFrame(similarity.cpu().numpy(), columns=sent['Sentence'].values)
    sim_df['Sentence'] = sent['Sentence'].values
    sim_df = sim_df.melt('Sentence', var_name='Sentence2', value_name='Similarity')
    sim_df = sim_df[~(sim_df.Sentence == sim_df.Sentence2)]

    sim_df['Sentence'] = sim_df['Sentence'].apply(lambda x: sent_tech_map[x])
    sim_df['Sentence2'] = sim_df['Sentence2'].apply(lambda x: sent_tech_map[x])
    sim_df['Similarity'] = sim_df['Similarity'].apply(lambda x: 0 if x < 0 else x)

    sim_mean_df = sim_df.groupby(["Sentence", "Sentence2"]).mean().reset_index()
    sim_mean_df = sim_mean_df.pivot('Sentence', 'Sentence2', "Similarity")

    plt.figure(figsize=(4, 3))
    sns.heatmap(sim_mean_df, annot=True, cmap="YlGnBu")
    plt.title("Cos-sim between Sentences (Mean)")
    plt.show()
    

def plot_sent_vs_desc(sent, tech, similarity):
    """
    Plot heatmap of similarity between sentences and technique descriptions
    
    :param sent: `pd.DataFrame` containing sentences (must include `Sentence`, `Sent. No.` and `Technique` columns)
    :param sent: `pd.DataFrame` containing technique descriptions (must include `Sent. No.` columns)
    :param similarity: Similarity matrix
    """
    column_index = tech.index.astype('str') + "_" + tech['Sent. No.'].astype('str')
    sim_df = pd.DataFrame(similarity.cpu().numpy(), columns=column_index)
    sim_df['Sentence'] =  sent['Technique']
    sim_df['Sent. No.'] = sent['Sent. No.']
    sim_df = sim_df.melt(['Sentence', 'Sent. No.'], var_name='Description', value_name='Similarity')
    sim_df['Similarity'] = sim_df['Similarity'].apply(lambda x: 0 if x < 0 else x)
    sim_df[['Description', 'Tech. Sent. No.']] = sim_df['Description'].str.split('_', 1, expand=True)
    sim_df['Tech. Sent. No.'] = sim_df['Tech. Sent. No.'].astype('int')
    
    sim_mean_df = sim_df.groupby(["Sentence", "Description"]).mean().reset_index()
    sim_mean_df = sim_mean_df.pivot('Sentence', 'Description', "Similarity")

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_mean_df, annot=True, cmap="YlGnBu")
    plt.title("Cos-sim between Sentence vs Description Sentences (Mean)")
    plt.show()