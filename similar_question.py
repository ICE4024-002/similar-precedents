import torch
from similarity import cal_cosine

def get_similar_question(question_vector, result_embeddings):
    prev_question_embeddings = torch.tensor([row[1] for row in result_embeddings])
    similarities = cal_cosine(question_vector.unsqueeze(0), prev_question_embeddings)

    max_similarity_idx = similarities.argmax().item()
    max_similarity = similarities[0][max_similarity_idx].item()

    question_id = result_embeddings[max_similarity_idx][0]

    return question_id, max_similarity