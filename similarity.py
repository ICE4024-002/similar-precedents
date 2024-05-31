import torch
from embedding import create_embeddings

# 코사인 유사도
def cal_cosine(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

# 유클리디안 유사도
def cal_euclidean(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    euclidean_distance = torch.cdist(a, b, p=2).item()
    
    max_distance = torch.sqrt(torch.tensor(a.shape[1])).item()
    normalized_distance = euclidean_distance / max_distance
    
    # Convert to similarity: smaller distance -> higher similarity
    return (1 - normalized_distance) * 100

# 피어슨 유사도
def cal_pearson(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)
    
    # Compute Pearson correlation
    a_mean = a - a.mean(dim=1, keepdim=True)
    b_mean = b - b.mean(dim=1, keepdim=True)
    
    numerator = (a_mean * b_mean).sum(dim=1)
    denominator = torch.sqrt((a_mean ** 2).sum(dim=1) * (b_mean ** 2).sum(dim=1))
    
    pearson_similarity = numerator / denominator
    
    return (pearson_similarity + 1) / 2 * 100

# 자카드 유사도
def cal_jaccard(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    jaccard_similarity = intersection / union
    
    
    return jaccard_similarity * 100

sentences = [
    ('한 남자가 음식을 먹고 있다.', '한 남자가 뭔가를 먹고 있다.'),
    ('한 비행기가 착륙하고 있다.', '애니메이션화된 비행기 하나가 착륙하고 있다.'),
    ('한 여성이 고기를 요리하고 있다.', '한 남자가 말하고 있다.')
    ]

senetece_vectors = [(create_embeddings(sen[0]), create_embeddings(sen[1])) for sen in sentences]

for i in range(0, 3):
    print(f'문장 쌍: {sentences[i]}')
    print(f'코사인 유사도: {cal_cosine(senetece_vectors[i][0], senetece_vectors[i][1]).item()}')
    print(f'유클리디안 유사도: {cal_euclidean(senetece_vectors[i][0], senetece_vectors[i][1])}')
    print(f'피어슨 유사도: {cal_pearson(senetece_vectors[i][0], senetece_vectors[i][1]).item()}')
    print(f'자카드 유사도: {cal_jaccard(sentences[i][0], sentences[i][1])}')
    print()