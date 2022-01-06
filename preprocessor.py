# _*_ coding: UTF-8 _*_
# Author LBK
# Desciption: 对数据进行预处理。 清洗， 转换为问答pair， 并保存。

import logging
import re
import sys
import os
import pathlib

import numpy as np
import pandas as pd

from config import dev_raw, root_path, sep, test_raw, train_raw
import config

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)



if __name__ == "__main__":
    dev =