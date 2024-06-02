import torch

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