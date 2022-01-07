# _*_ coding: UTF-8 _*_
# Author LBK
import logging
import sys
import os

import pandas as pd
import fasttext
from tqdm import tqdm
import jieba.posseg as pseg

# sys.path.append("..")
import config
from config import root_path
from preprocessor import clean, filter_content

tqdm.pandas()

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class Intention(object):
    """
    建立fasttext 模型， 判断用户输入是否属于业务咨询。
    """
    def __init__(self,
                 data_path=config.train_path,
                 sku_path=config.ware_path,
                 model_path=None,
                 kw_path=None,
                 model_train_file=config.business_train,
                 model_test_file=config.business_test):
        self.model_path = model_path
        self.data = pd.read_csv(data_path)

        if model_path and os.path.exists(model_path):
            self.fast = fasttext.load_model(model_path)
        else:
            self.kw = self.build_keyword(sku_path, to_file=kw_path)
            self.data_process(model_train_file)
            # self.fast = self.train(model_train_file, model_test_file)

    def build_keyword(self, sku_path, to_file):
        """
        构建业务咨询相关关键词, 并保存
        :param sku_path: JD sku 文件路径
        :param to_file: 关键词保存路径
        :return: 关键词list
        """
        logging.info("Building keywords.")
        tokens = []

        # Filtering words according to POS tags.
        tokens = self.data["custom"].dropna().apply(lambda x: [token for token, pos in pseg.cut(x) if pos in ["n", "vn", "nz"]])
        key_words = set([tk for idx, sample in tokens.iteritems() for tk in sample if len(tk) > 1])

        logging.info('Key words built.')

        sku = []
        with open(sku_path, "r") as f:
            next(f)
            for lines in f:
                line = lines.strip().split("\t")
                sku.extend(line[-1].split("/"))

        key_words |= set(sku)
        logging.info("Sku words merged.")

        if to_file:
            with open(to_file, "w") as f:
                for i in key_words:
                    f.write(i + "\n")

        return key_words

    def data_process(self, model_train_file):
        """
        判断咨询中是否包含业务关键词, 如果包含 label 为 1, 否则为 0
        并处理成 fasttext 需要的数据格式
        :param model_train_file: 模型训练数据保存路径
        :return:
        """
        logging.info("Process data.")
        self.data["is_business"] = self.data["custom"].progress_apply(
            lambda x: 1 if any(kw in x for kw in self.kw) else 0
        )

        with open(model_train_file, "w") as f:
            for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                outline = clean(row["custom"]) + "\t__label__" + str(int(row["is_business"])) + "\n"
                f.write(outline)


if __name__ == "__main__":
    it = Intention(config.train_path,
                   config.ware_path,
                   model_path=config.ft_path,
                   kw_path=config.keyword_path)
    # print(it.predict("怎么申请价保呢?"))
    # print(it.predict("怎么申请价保呢?"))
