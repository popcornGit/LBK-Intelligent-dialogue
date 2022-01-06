# _*_ coding: UTF-8 _*_
# Author LBK
# Description: Combine intention module, retrieval module and ranking module for task-oriented dialogue.

import os

import pandas as pd
import config
from intention.business import Intention
from retrieval.hnsw_faiss import HNSW
from ranking.ranker import RANK


def retrieve(k):
    pass


def rank():
    pass


if __name__ == "__main__":
    retrieve(5)
    rank()
