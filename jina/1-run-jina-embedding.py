from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the model
model = AutoModel.from_pretrained("/Users/randy/Downloads/models/jinaai/jina-embeddings-v3", trust_remote_code=True)

model.to('mps')

texts = [
    "Follow the white rabbit.",  # English
    "Sigue al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
]

# When calling the `encode` function, you can choose a `task` based on the use case:
# 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
# Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
embeddings = model.encode(texts, task="retrieval.query")

# Compute similarities
ret = cosine_similarity(embeddings, embeddings)

print(ret)