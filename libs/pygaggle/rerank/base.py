from typing import Union, Optional, Mapping, Any
from copy import deepcopy
import abc

__all__ = ['Query', 'Text', 'Reranker', 'TextType']


TextType = Union['Query', 'Text']


class Query:
    """Class representing a query.
    A query contains the query text itself and potentially other metadata.

    Parameters
    ----------
    text : str
        The query text.
    id : Optional[str]
        The query id.
    """

    def __init__(self, text: str, id: Optional[str] = None, metadata: Mapping[str, Any] = None):
        self.text = text
        self.id = id
        if metadata is None:
            metadata = dict()
        self.metadata = metadata


class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.

    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The text's title.
    """

    def __init__(self,
                 text: str,
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 title: Optional[str] = None):
        self.text = text
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.title = title

from libs.pygaggle.data.segmentation import SegmentProcessor

class Reranker:
    """Class representing a reranker.
    A reranker takes a list texts and returns a list of texts non-destructively
    (i.e., does not alter the original input list of texts).
    """

    def rerank(self, query: Query, texts: list[Text]) -> list[Text]:
        """Sorts a list of texts
        """
        return sorted(self.rescore(query, texts), key=lambda x: x.score, reverse=True)

    def rerank_by_segment(self, query: Query, texts: list[Text], seg_size: int = 8, 
                          stride: int = 6, aggregate_method: str = 'max') -> list[Text]:
        """Sorts a list of texts by segments
        """
        segment_processor = SegmentProcessor()          
        segment_group = segment_processor.segment(texts, seg_size, stride)
        segment_group.segments = self.rescore(query, segment_group.segments)
        docs = segment_processor.aggregate(texts, segment_group, aggregate_method)
                              
        return sorted(docs, key=lambda x: x.score, reverse=True)

    @abc.abstractmethod
    def rescore(self, query: Query, texts: list[Text]) -> list[Text]:
        """Reranks a list of texts with respect to a query.

         Parameters
         ----------
         query : Query
             The query.
         texts : list[Text]
             The list of texts.

         Returns
         -------
         list[Text]
             Reranked list of texts.
         """
        pass
