from dataclasses import dataclass
from typing import Optional

from libs.pygaggle.rerank.base import Query, Text

__all__ = ['RelevanceExample']


@dataclass
class RelevanceExample:
    query: Query
    documents: list[Text]
    labels: list[bool]
