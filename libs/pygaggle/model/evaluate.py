from collections import OrderedDict
from typing import Optional
from copy import deepcopy
from pathlib import Path
import os
import abc

from sklearn.metrics import recall_score
from tqdm import tqdm
import numpy as np
import string
import regex as re

from libs.pygaggle.data.relevance import RelevanceExample
from libs.pygaggle.rerank.base import Reranker
from libs.pygaggle.model.writer import Writer, SimpleWriter
from libs.pygaggle.data.segmentation import SegmentProcessor


__all__ = ['RerankerEvaluator', 'DuoRerankerEvaluator', 'StepEvaluator', 'MultiStageEvaluator', 'metric_names']
METRIC_MAP = OrderedDict()


class MetricAccumulator:
    name: str = None

    def accumulate(self, scores: list[float], gold: list[RelevanceExample]):
        return

    @abc.abstractmethod
    def value(self):
        return


class MeanAccumulator(MetricAccumulator):
    def __init__(self):
        self.scores = []

    @property
    def value(self):
        return np.mean(self.scores)


class TruncatingMixin:
    def truncated_rels(self, scores: list[float]) -> np.ndarray:
        return np.array(scores)


def register_metric(name):
    def wrap_fn(metric_cls):
        METRIC_MAP[name] = metric_cls
        metric_cls.name = name
        return metric_cls
    return wrap_fn


def metric_names():
    return list(METRIC_MAP.keys())


class TopkMixin(TruncatingMixin):
    top_k: int = None

    def truncated_rels(self, scores: list[float]) -> np.ndarray:
        rel_idxs = sorted(list(enumerate(scores)),
                          key=lambda x: x[1], reverse=True)[self.top_k:]
        scores = np.array(scores)
        scores[[x[0] for x in rel_idxs]] = -1
        scores[scores == -np.inf] = -1
        return scores


class DynamicThresholdingMixin(TruncatingMixin):
    threshold: float = 0.5

    def truncated_rels(self, scores: list[float]) -> np.ndarray:
        scores = np.array(scores)
        scores[scores < self.threshold * np.max(scores)] = 0
        return scores


class RecallAccumulator(TruncatingMixin, MeanAccumulator):
    def accumulate(self, scores: list[float], gold: RelevanceExample):
        score_rels = self.truncated_rels(scores)
        score_rels[score_rels != -1] = 1
        score_rels[score_rels == -1] = 0
        gold_rels = np.array(gold.labels, dtype=int)
        score = recall_score(gold_rels, score_rels, zero_division=0)
        self.scores.append(score)


class PrecisionAccumulator(TruncatingMixin, MeanAccumulator):
    def accumulate(self, scores: list[float], gold: RelevanceExample):
        score_rels = self.truncated_rels(scores)
        score_rels[score_rels != -1] = 1
        score_rels[score_rels == -1] = 0
        score_rels = score_rels.astype(int)
        gold_rels = np.array(gold.labels, dtype=int)
        sum_score = score_rels.sum()
        if sum_score > 0:
            self.scores.append((score_rels & gold_rels).sum() / sum_score)


@register_metric('precision@1')
class PrecisionAt1Metric(TopkMixin, PrecisionAccumulator):
    top_k = 1

    
@register_metric('recall@2')
class RecallAt2Metric(TopkMixin, RecallAccumulator):
    top_k = 2

    
@register_metric('recall@3')
class RecallAt3Metric(TopkMixin, RecallAccumulator):
    top_k = 3

    
@register_metric('recall@5')
class RecallAt5Metric(TopkMixin, RecallAccumulator):
    top_k = 5
    

@register_metric('recall@10')
class RecallAt10Metric(TopkMixin, RecallAccumulator):
    top_k = 10

    
@register_metric('recall@20')
class RecallAt20Metric(TopkMixin, RecallAccumulator):
    top_k = 20

    
@register_metric('recall@40')
class RecallAt40Metric(TopkMixin, RecallAccumulator):
    top_k = 40


@register_metric('recall@50')
class RecallAt50Metric(TopkMixin, RecallAccumulator):
    top_k = 50


@register_metric('recall@100')
class RecallAt100Metric(TopkMixin, RecallAccumulator):
    top_k = 100


@register_metric('mrr')
class MrrMetric(MeanAccumulator):
    def accumulate(self, scores: list[float], gold: RelevanceExample):
        scores = sorted(list(enumerate(scores)),
                        key=lambda x: x[1], reverse=True)
        rr = next((1 / (rank_idx + 1) for rank_idx, (idx, _) in
                   enumerate(scores) if gold.labels[idx]), 0)
        self.scores.append(rr)


@register_metric('mrr@10')
class MrrAt10Metric(MeanAccumulator):
    def accumulate(self, scores: list[float], gold: RelevanceExample):
        scores = sorted(list(enumerate(scores)), key=lambda x: x[1],
                        reverse=True)
        rr = next((1 / (rank_idx + 1) for rank_idx, (idx, _) in
                   enumerate(scores) if (gold.labels[idx] and rank_idx < 10)),
                  0)
        self.scores.append(rr)


class ThresholdedRecallMetric(DynamicThresholdingMixin, RecallAccumulator):
    threshold = 0.5


class ThresholdedPrecisionMetric(DynamicThresholdingMixin,
                                 PrecisionAccumulator):
    threshold = 0.5


class RerankerEvaluator:
    def __init__(self,
                 reranker: Reranker,
                 metrics: list[str],
                 use_tqdm: bool = True,
                 writer: Optional[Writer] = None):
        self.reranker = reranker
        self.metrics = [METRIC_MAP[name] for name in metrics]
        self.use_tqdm = use_tqdm
        self.writer = writer

    def evaluate(self,
                 examples: list[RelevanceExample]) -> list[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        for example in tqdm(examples, disable=not self.use_tqdm):
            scores = [x.score for x in self.reranker.rescore(example.query,
                                                             example.documents)]
            if self.writer is not None:
                self.writer.write(scores, example)
            for metric in metrics:
                metric.accumulate(scores, example)
        return metrics

    def evaluate_by_segments(self,
                             examples: list[RelevanceExample],
                             seg_size: int,
                             stride: int,
                             aggregate_method: Optional[str] = None) -> list[MetricAccumulator]:
        
        metrics = [cls() for cls in self.metrics]
        segment_processor = SegmentProcessor()
        for example in tqdm(examples, disable=not self.use_tqdm):
            segment_group = segment_processor.segment(example.documents, seg_size, stride)
            segment_group.segments = self.reranker.rescore(example.query, segment_group.segments)
            doc_scores = [x.score for x in segment_processor.aggregate(example.documents,
                                                                       segment_group,
                                                                       aggregate_method)]
            if self.writer is not None:
                self.writer.write(doc_scores, example)
            for metric in metrics:
                metric.accumulate(doc_scores, example)
        return metrics
    

class StepEvaluator:
    def __init__(self,
                 reranker: Reranker,
                 metrics: list[str],
                 n_hits: int = 50,
                 use_tqdm: bool = True):
        self.reranker = reranker
        self.n_hits = n_hits
        self.metrics = [METRIC_MAP[name] for name in metrics]
        self.use_tqdm = use_tqdm

    def evaluate(self,
                 examples: list[RelevanceExample]) -> tuple[list[RelevanceExample], list[MetricAccumulator]]:
        metrics = [cls() for cls in self.metrics]
        out_examples = []
        for ct, example in tqdm(enumerate(examples), total=len(examples), disable=not self.use_tqdm):
            doc_scores = self.reranker.rescore(example.query, example.documents)
            scores = np.array([x.score for x in doc_scores])
            
            for metric in metrics:
                metric.accumulate(list(scores), example)
              
            filtered = sorted(enumerate(doc_scores), key=lambda x: x[1].score, reverse=True)[:self.n_hits]
            filtered_labels = np.array(example.labels)[list(map(lambda x: x[0], filtered))]
            out_examples.append(RelevanceExample(
                example.query,
                list(map(lambda x: x[1], filtered)),
                list(filtered_labels)
            ))
            
        return out_examples, metrics
    
    def evaluate_by_segments(self,
                             examples: list[RelevanceExample],
                             seg_size: int,
                             stride: int,
                             aggregate_method: str) -> list[MetricAccumulator]:
        
        metrics = [cls() for cls in self.metrics]
        out_examples = []
        segment_processor = SegmentProcessor()
        for example in tqdm(examples, disable=not self.use_tqdm):
            segment_group = segment_processor.segment(example.documents, seg_size, stride)
            segment_group.segments = self.reranker.rescore(example.query, segment_group.segments)
            doc_scores = segment_processor.aggregate(example.documents,
                                                     segment_group,
                                                     aggregate_method)
            scores = np.array([x.score for x in doc_scores])
            
            for metric in metrics:
                metric.accumulate(list(scores), example)
                
            filtered = sorted(enumerate(doc_scores), key=lambda x: x[1].score, reverse=True)[:self.n_hits]
            filtered_labels = np.array(example.labels)[list(map(lambda x: x[0], filtered))]
            out_examples.append(RelevanceExample(
                example.query,
                list(map(lambda x: x[1], filtered)),
                list(filtered_labels)
            ))
        
        return out_examples, metrics


class MultiStageEvaluator:
    def __init__(self,
                 rerankers: list[Reranker],
                 metrics: list[str],
                 n_hits: list[int],
                 use_tqdm: bool = True):
        assert len(rerankers) == len(n_hits), "Length of `rerankers` and `n_hits` should be equal."
        
        self.rerankers = rerankers
        self.n_hits = n_hits
        self.metrics = [METRIC_MAP[name] for name in metrics]
        self.use_tqdm = use_tqdm

    def evaluate(self,
                 examples: list[RelevanceExample]) -> list[MetricAccumulator]:
        metrics = [[cls() for cls in self.metrics] for _ in range(len(self.rerankers))]
        stage_texts = []
        
        for ct, example in tqdm(enumerate(examples), total=len(examples), disable=not self.use_tqdm):
            out = self.rerankers[0].rescore(example.query, example.documents)
            stage_texts.append(sorted(enumerate(out), key=lambda x: x[1].score, reverse=True)[:self.n_hits[0]])
            score = np.array([x.score for x in out])
            for metric in metrics[0]:
                metric.accumulate(list(score), example)
            
        for i, reranker in enumerate(self.rerankers[1:] if len(self.rerankers) > 1 else []):
            current_texts = deepcopy(stage_texts)
            stage_texts = []
            for ct, texts in tqdm(enumerate(current_texts), total=len(current_texts), disable=not self.use_tqdm):
                keys = list(map(lambda x: x[0], texts))
                stage_in = list(map(lambda x: x[1], texts))
                
                out = reranker.rescore(examples[ct].query, stage_in)
                stage_texts.append(sorted(zip(keys, out), key=lambda x: x[1].score, reverse=True)[:self.n_hits[i+1]])
                score = np.array([x.score for x in out])

                stage_scores = np.ones(len(examples[ct].labels)) * -np.inf
                stage_scores[keys] = score

                for metric in metrics[i+1]:
                    metric.accumulate(list(stage_scores), examples[ct])
        return metrics


class DuoRerankerEvaluator:
    def __init__(self,
                 mono_reranker: Reranker,
                 duo_reranker: Reranker,
                 metrics: list[str],
                 mono_hits: int = 50,
                 use_tqdm: bool = True,
                 writer: Optional[Writer] = None,
                 mono_cache_write_path: Optional[Path] = None,
                 skip_mono: bool = False):
        self.mono_reranker = mono_reranker
        self.duo_reranker = duo_reranker
        self.mono_hits = mono_hits
        self.metrics = [METRIC_MAP[name] for name in metrics]
        self.use_tqdm = use_tqdm
        self.writer = writer
        self.mono_cache_writer = None
        self.skip_mono = skip_mono
        if not self.skip_mono:
            self.mono_cache_writer = SimpleWriter(mono_cache_write_path)

    def evaluate(self,
                 examples: list[RelevanceExample]) -> list[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        mono_texts = []
        scores = []
        if not self.skip_mono:
            for ct, example in tqdm(enumerate(examples), total=len(examples), disable=not self.use_tqdm):
                mono_out = self.mono_reranker.rescore(example.query, example.documents)
                mono_texts.append(sorted(enumerate(mono_out), key=lambda x: x[1].score, reverse=True)[:self.mono_hits])
                scores.append(np.array([x.score for x in mono_out]))
                if self.mono_cache_writer is not None:
                    self.mono_cache_writer.write(list(scores[ct]), examples[ct])
        else:
            for ct, example in tqdm(enumerate(examples), total=len(examples), disable=not self.use_tqdm):
                mono_out = example.documents
                mono_texts.append(list(enumerate(mono_out))[:self.mono_hits])
                scores.append(np.array([float(x.score) for x in mono_out]))
        for ct, texts in tqdm(enumerate(mono_texts), total=len(mono_texts), disable=not self.use_tqdm):
            duo_in = list(map(lambda x: x[1], texts))
            duo_scores = [x.score for x in self.duo_reranker.rescore(examples[ct].query, duo_in)]

            scores[ct][list(map(lambda x: x[0], texts))] = duo_scores
            if self.writer is not None:
                self.writer.write(list(scores[ct]), examples[ct])
            for metric in metrics:
                metric.accumulate(list(scores[ct]), examples[ct])
        return metrics

    def evaluate_by_segments(self,
                             examples: list[RelevanceExample],
                             seg_size: int,
                             stride: int,
                             aggregate_method: str) -> list[MetricAccumulator]:
        
        metrics = [cls() for cls in self.metrics]
        segment_processor = SegmentProcessor()
        for example in tqdm(examples, disable=not self.use_tqdm):
            segment_group = segment_processor.segment(example.documents, seg_size, stride)
            segment_group.segments = self.reranker.rescore(example.query, segment_group.segments)
            doc_scores = [x.score for x in segment_processor.aggregate(example.documents,
                                                                       segment_group,
                                                                       aggregate_method)]
            if self.writer is not None:
                self.writer.write(doc_scores, example)
            for metric in metrics:
                metric.accumulate(doc_scores, example)
        return metrics
