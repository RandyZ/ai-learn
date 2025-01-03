from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# 初始化模型
model = AutoModel.from_pretrained("/Users/randy/Downloads/models/jinaai/jina-embeddings-v3", trust_remote_code=True)
model.to('mps')

texts = [
    "Follow the white rabbit.",  # English
    "Sigue al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
    "在前面那只白色的兔子后面走。",  # Similar Chinese
]

# When calling the `encode` function, you can choose a `task` based on the use case:
# 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
# Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
# 定义所有任务类型
tasks = ['retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching']

# 存储每个任务的相似度矩阵
similarities = {}

# 计算每个任务的嵌入向量和相似度
for task in tasks:
    embeddings = model.encode(texts, task=task)
    sim_matrix = cosine_similarity(embeddings, embeddings)
    similarities[task] = sim_matrix

# 打印结果
languages = ['English', 'Spanish', 'French', 'Chinese', 'Arabic', 'German', 'Similar']

for task in tasks:
    print(f"\n=== 任务模式: {task} ===")
    df = pd.DataFrame(
        similarities[task], 
        index=languages,
        columns=languages
    )
    print(df.round(3))

# 计算任务间的差异
print("\n=== 不同任务模式之间的平均差异 ===")
task_differences = {}
for i in range(len(tasks)):
    for j in range(i+1, len(tasks)):
        task1, task2 = tasks[i], tasks[j]
        diff = np.abs(similarities[task1] - similarities[task2]).mean()
        task_differences[f"{task1} vs {task2}"] = diff

for pair, diff in task_differences.items():
    print(f"{pair}: {diff:.3f}")