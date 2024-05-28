import torch

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

def find_similar_precedent(question_vector, data, result_embeddings):
    similarities = cal_score(question_vector.unsqueeze(0), result_embeddings)
    max_similarity = similarities.max().item()
    max_similarity_idx = similarities.argmax().item()

    return data[max_similarity_idx], max_similarity

def get_similar_precedent_total(data, result_embeddings, question_vector):
    similarities = cal_score(question_vector.unsqueeze(0), result_embeddings)
    max_similarity = similarities.max().item()
    max_similarity_idx = similarities.argmax().item()

    return data[max_similarity_idx], max_similarity

# 유사한 질문 및 전문가 평가를 위한 쿼리
# evaluation_query = text(f"""
#                         """)
# evaluation_result = connection.execute(evaluation_query).fetchall()

# evaluation_result_embeddings = torch.tensor([
#     [float(num_str) for num_str in row[0][1:-1].split(',')]
#     for row in evaluation_result
# ])

# questions = [row[1] for row in evaluation_result]
# expert_evaluations = [row[2] for row in evaluation_result]

# 질문의 유사도를 계산해서 가장 높은 유사도를 가진 질문과 전문가 평가를 반환
# def get_most_similar_question(question):
#     encoded_input = tokenizer([question], padding=True, truncation=True, return_tensors='pt')

#     with torch.no_grad():
#         model_output = model(**encoded_input)
    
#     embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

#     question_vector = embeddings[0]

#     similarities = cal_score(question_vector.unsqueeze(0), evaluation_result_embeddings)
#     max_similarity = similarities.max().item()
#     max_similarity_idx = similarities.argmax().item()

#     most_similar_question = questions[max_similarity_idx]
#     most_similar_evaluation = evaluation_result_embeddings[max_similarity_idx]

#     if max_similarity < 80:
#         return None, None, None
#     return most_similar_question, most_similar_evaluation, max_similarity