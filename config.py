# _*_ coding: UTF-8 _*_
# Author LBK
# Description: 配置文件

import torch
import os

root_path = os.path.abspath(os.path.dirname(__file__))

train_raw = os.path.join(root_path, "data/chat.txt")
dev_raw = os.path.join(root_path, "data/Develop.txt")
test_raw = os.path.join(root_path, "data/Test.txt")
ware_path = os.path.join(root_path, "data/ware.txt")

sep = "[SEP]"

"""Data"""
# main
train_path = os.path.join(root_path, "data/train_no_blank.csv")
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')

# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')

"""Intention"""
# fasttext
ft_path = os.path.join(root_path, "model/intention/fastext")

"""Retrieval"""
# Embedding
w2v_path = os.path.join(root_path, "model/retrieval/word2vec")

# HNSW parameters
ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
M = 64  # M defines the maximum number of outgoing connections in the graph
hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw_index')

# 通用配置
is_cuda = True
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
