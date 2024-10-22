from config import *
from tool import *

# query_template = """\
# 根据文本产生尽可能多样化的提问。提问中包含明确的上下文信息，问题答案必须包含在文本中。一共产生60个提问。
# 文本：
# {}

# 输出格式为JSON字符串列表
# ```json
# [
#     "1. 问题1",
#     "2. 问题2",
#     ...
# ]
# ```
# """
# Who, What, When, Where, Why, How, Effect, How many, Knowledge Source等
# 你是一位检索增强生成系统管理员，请生成以下文本能准确支持回答的问题。
# 要求1：问题必须能被文本明确支持回答，避免过于发散。
# 要求2：问题包含明确的上下文信息，避免指代模糊。
# 要求3：共产生20个提问。提问内容允许重复，但形式应灵活多变。
query_template = """\
你是一位检索增强生成系统管理员，请生成以下文本能准确支持回答的问题。
要求1：问题必须能**被文本信息明确支持回答**。
要求2：问题包含明确详细的上下文信息，避免指代模糊，提问对象预先不知道内容，需要根据你的提问查找文本
要求3：共产生60个提问。提问内容允许重复，但形式应灵活多变。
示例：在最近发表在《自然》杂志上的癌症研究中，名为NP137的治疗性单克隆抗体在人体试验中对多少名晚期子宫内膜癌患者显示出了抗肿瘤响应？

文本：
{}

输出格式为JSON字符串列表
```json
[
    "1. 问题",
    "2. 问题",
    ...
]
```
"""


def get_questions(chunk):
   query = query_template.format(chunk)
   return get_chat_completion(query)

if __name__ == '__main__':
    chunk = "研究团队记录了参与者的坚果食用量，包括无盐杏仁、腰果、开心果、腌或烤坚果和花生。参与者接受了5年时间的随访，在此期间，8%的参与者被诊断出罹患了抑郁症。分析显示，与不食用坚果的人相比，低至中度食用坚果（相当于每天30克）可降低17%的抑郁症风险。一份30克的坚果大约相当于20颗杏仁、10颗巴西坚果、15颗腰果、40颗花生或30颗开心果。"
    res, chat = get_questions(chunk)
    print(res)
    # print(chat)

    # from sentence_transformers import SentenceTransformer
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import warnings
    # warnings.filterwarnings("ignore")

    # model = SentenceTransformer("/mnt/data102_d2/huggingface/models/bge-base-zh-v1.5/")
    # coeff = 1
    # docs = [doc.split('. ')[1] for doc in res]
    # docs = [chunk] + docs
    # doc_embeddings = model.encode(docs, normalize_embeddings=True)

