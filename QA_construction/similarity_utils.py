import jieba
import evaluate
from sentence_transformers import util

def tokenize(text):
    return list(jieba.cut(text))


# 计算 BLEU 分数
def calculate_bleu(candidate, reference, with_penalty=False):
    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=[candidate], references=[[reference]], tokenizer=tokenize)

    bleu_avg = results['bleu']
    bleu1 = results['precisions'][0]
    bleu2 = results['precisions'][1]
    bleu3 = results['precisions'][2]
    bleu4 = results['precisions'][3]
    brevity_penalty = results['brevity_penalty']

    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    else:
        return 0.0 if brevity_penalty == 0 else bleu_avg / brevity_penalty, bleu1, bleu2, bleu3, bleu4


# 计算 ROUGE-L 分数
def rougeL_score(continuation: str, reference: str) -> float:
    f = lambda text: list(jieba.cut(text))
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=[continuation], references=[[reference]], tokenizer=f, rouge_types=['rougeL'])
    score = results['rougeL']  # 提取 ROUGE-L 分数（
    return score


# 计算 Jaccard 相似度
def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


# 计算 BGE 相似度
def bge_similarity(text1, text2, model):
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    return util.pytorch_cos_sim(embedding1, embedding2).item()


if __name__ == '__main__':
    # 加载 BAAI/bge-base-en-v1.5 模型
    # model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    # 输入文本
    textA = "Festival Cruises had an option for two more ships of the enlarged Mistral design, but the company decided not to use the option. Two more Mistral class ships were however built for MSC Cruises as MSC Lirica and MSC Opera."
    textB = "Lirica"
    textC = "Mistral-class cruise ship"

    # 计算相似度
    jaccard_AB = jaccard_similarity(textA, textB)
    jaccard_AC = jaccard_similarity(textA, textC)

    bge_AB = bge_similarity(textA, textB)
    bge_AC = bge_similarity(textA, textC)

    # 输出结果
    print(f"BGE Similarity between A and B: {bge_AB:.4f}")
    print(f"BGE Similarity between A and C: {bge_AC:.4f}")
    print(f"Jaccard Similarity between A and B: {jaccard_AB:.4f}")
    print(f"Jaccard Similarity between A and C: {jaccard_AC:.4f}")