import torch
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from similarity import cal_cosine, cal_euclidean, cal_pearson, cal_jaccard

# 유사도 계산 잘 되는지 확인하는 예제
# sentences = [
#     ('한 남자가 음식을 먹고 있다.', '한 남자가 뭔가를 먹고 있다.'),
#     ('한 비행기가 착륙하고 있다.', '애니메이션화된 비행기 하나가 착륙하고 있다.'),
#     ('한 여성이 고기를 요리하고 있다.', '한 남자가 말하고 있다.')
#     ]

# senetece_vectors = [(create_embeddings(sen[0]), create_embeddings(sen[1])) for sen in sentences]

# for i in range(0, 3):
#     print(f'문장 쌍: {sentences[i]}')
#     print(f'코사인 유사도: {cal_cosine(senetece_vectors[i][0], senetece_vectors[i][1]).item()}')
#     print(f'유클리디안 유사도: {cal_euclidean(senetece_vectors[i][0], senetece_vectors[i][1])}')
#     print(f'피어슨 유사도: {cal_pearson(senetece_vectors[i][0], senetece_vectors[i][1]).item()}')
#     print(f'자카드 유사도: {cal_jaccard(sentences[i][0], sentences[i][1])}')
#     print()

# 각각 유사도 사용했을 때 Label(0-5)이랑 얼마나 유사한지 확인
# 비교를 위해 Label을 0-100으로 정규화
# 유사도 여러 개를 사용할 때는 어떻게 최종 유사도 계산?
    # 1. 평균
    # 2. 최솟값
# 각 데이터별로 Label과 최종 유사도의 차이를 계산 후 평균을 내어 성능을 평가
# Label과 최종 유사도의 차이가 적을수록 좋은 성능을 보이는 유사도 조합이라 판단

# KorSTS 데이터셋 로드
sts_total = pd.read_csv('./korsts/sts-total.tsv', delimiter='\t')
sts_total['percent_score'] = sts_total['score'] * 20

# KorSTS 데이터셋 벡터 로드
embedded_sentence1 = pd.read_csv('./korsts/embedded_sentence1.csv')
embedded_sentence2 = pd.read_csv('./korsts/embedded_sentence2.csv')

sentence1_vectors = []
sentence2_vectors = []
for i in range(0, len(embedded_sentence1)):
    arr = []
    for elem in embedded_sentence1.iloc[i]:
        arr.append(elem)
    sentence1_vectors.append(arr)

for i in range(0, len(embedded_sentence2)):
    arr = []
    for elem in embedded_sentence2.iloc[i]:
        arr.append(elem)
    sentence2_vectors.append(arr)

sentence1_vectors = torch.tensor(sentence1_vectors)
sentence2_vectors = torch.tensor(sentence2_vectors)

print('>>> Sentence 1, 2 Vector loaded')

print('각 유사도별 Label과의 차이의 평균')
# 코사인 유사도
for i in range(0, len(sentence1_vectors)):
    cosine_similarity = cal_cosine(sentence1_vectors[i], sentence2_vectors[i]).item()
    sts_total.at[i, 'cosine_similarity'] = cosine_similarity
sts_total['cosine_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_similarity'])
cosine_diff_mean = sts_total.iloc[1:]['cosine_diff'].mean()

print("코사인 유사도: ", cosine_diff_mean)

# 유클리디안 유사도
for i in range(0, len(sentence1_vectors)):
    euclidean_similarity = cal_euclidean(sentence1_vectors[i], sentence2_vectors[i])
    sts_total.at[i, 'euclidean_similarity'] = euclidean_similarity
sts_total['euclidean_diff'] = abs(sts_total['percent_score'] - sts_total['euclidean_similarity'])
euclidean_diff_mean = sts_total.iloc[1:]['euclidean_diff'].mean()

print("유클리디안 유사도: ", euclidean_diff_mean)

# 피어슨 유사도
for i in range(0, len(sentence1_vectors)):
    pearson_similarity = cal_pearson(sentence1_vectors[i], sentence2_vectors[i]).item()
    sts_total.at[i, 'pearson_similarity'] = pearson_similarity
sts_total['pearson_diff'] = abs(sts_total['percent_score'] - sts_total['pearson_similarity'])
pearson_diff_mean = sts_total.iloc[1:]['pearson_diff'].mean()

print("피어슨 유사도: ", pearson_diff_mean)

# 자카드 유사도
for index, row in sts_total.iterrows():
    try:
        jaccard_similarity = cal_jaccard(row['sentence1'], row['sentence2'])
        sts_total.at[index, 'jaccard_similarity'] = jaccard_similarity
    except Exception as e:
        pass
sts_total['jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['jaccard_similarity'])
jaccard_diff_mean = sts_total.iloc[1:]['jaccard_diff'].mean()

print("자카드 유사도: ", jaccard_diff_mean)

print()

# 코사인 유사도 + 유클리디안 유사도
for elem in sts_total:
    sts_total['cosine_euclidean_similarity'] = sts_total.apply(lambda row: min(row['cosine_similarity'], row['euclidean_similarity']), axis=1)
sts_total['cosine_euclidean_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_euclidean_similarity'])
cosine_euclidean_diff_mean = sts_total.iloc[1:]['cosine_euclidean_diff'].mean()

print("코사인 유사도 + 유클리디안 유사도: ", cosine_euclidean_diff_mean)
# 코사인 유사도 + 피어슨 유사도
for elem in sts_total:
    sts_total['cosine_pearson_similarity'] = sts_total.apply(lambda row: min(row['cosine_similarity'], row['pearson_similarity']), axis=1)
sts_total['cosine_pearson_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_pearson_similarity'])
cosine_pearson_diff_mean = sts_total.iloc[1:]['cosine_pearson_diff'].mean()

print("코사인 유사도 + 피어슨 유사도: ", cosine_pearson_diff_mean)

# 코사인 유사도 + 자카드 유사도
for elem in sts_total:
    sts_total['cosine_jaccard_similarity'] = sts_total.apply(lambda row: min(row['cosine_similarity'], row['jaccard_similarity']), axis=1)
sts_total['cosine_jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_jaccard_similarity'])
cosine_jaccard_diff_mean = sts_total.iloc[1:]['cosine_jaccard_diff'].mean()

print("코사인 유사도 + 자카드 유사도: ", cosine_jaccard_diff_mean)

# 유클리디안 유사도 + 피어슨 유사도
for elem in sts_total:
    sts_total['euclidean_pearson_similarity'] = sts_total.apply(lambda row: min(row['euclidean_similarity'], row['pearson_similarity']), axis=1)
sts_total['euclidean_pearson_diff'] = abs(sts_total['percent_score'] - sts_total['euclidean_pearson_similarity'])
euclidean_pearson_diff_mean = sts_total.iloc[1:]['euclidean_pearson_diff'].mean()

print("유클리디안 유사도 + 피어슨 유사도: ", euclidean_pearson_diff_mean)

# 유클리디안 유사도 + 자카드 유사도
for elem in sts_total:
    sts_total['euclidean_jaccard_similarity'] = sts_total.apply(lambda row: min(row['euclidean_similarity'], row['jaccard_similarity']), axis=1)
sts_total['euclidean_jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['euclidean_jaccard_similarity'])
euclidean_jaccard_diff_mean = sts_total.iloc[1:]['euclidean_jaccard_diff'].mean()

print("유클리디안 유사도 + 자카드 유사도: ", euclidean_jaccard_diff_mean)

# 피어슨 유사도 + 자카드 유사도
for elem in sts_total:
    sts_total['pearson_jaccard_similarity'] = sts_total.apply(lambda row: min(row['pearson_similarity'], row['jaccard_similarity']), axis=1)
sts_total['pearson_jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['pearson_jaccard_similarity'])
pearson_jaccard_diff_mean = sts_total.iloc[1:]['pearson_jaccard_diff'].mean()

print("피어슨 유사도 + 자카드 유사도: ", pearson_jaccard_diff_mean)

print()

# 코사인 유사도 + 유클리디안 유사도 + 피어슨 유사도
for elem in sts_total:
    sts_total['cosine_euclidean_pearson_similarity'] = sts_total.apply(lambda row: min(row['cosine_similarity'], row['euclidean_similarity'], row['pearson_similarity']), axis=1)
sts_total['cosine_euclidean_pearson_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_euclidean_pearson_similarity'])
cosine_euclidean_pearson_diff_mean = sts_total.iloc[1:]['cosine_euclidean_pearson_diff'].mean()

# 코사인 유사도 + 유클리디안 유사도 + 자카드 유사도
for elem in sts_total:
    sts_total['cosine_euclidean_jaccard_similarity'] = sts_total.apply(lambda row: min(row['cosine_similarity'], row['euclidean_similarity'], row['jaccard_similarity']), axis=1)
sts_total['cosine_euclidean_jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_euclidean_jaccard_similarity'])
cosine_euclidean_jaccard_diff_mean = sts_total.iloc[1:]['cosine_euclidean_jaccard_diff'].mean()

print("코사인 유사도 + 유클리디안 유사도 + 자카드 유사도: ", cosine_euclidean_jaccard_diff_mean)
# 코사인 유사도 + 피어슨 유사도 + 자카드 유사도
for elem in sts_total:
    sts_total['cosine_pearson_jaccard_similarity'] = sts_total.apply(lambda row: min(row['cosine_similarity'], row['pearson_similarity'], row['jaccard_similarity']), axis=1)
sts_total['cosine_pearson_jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_pearson_jaccard_similarity'])
cosine_pearson_jaccard_diff_mean = sts_total.iloc[1:]['cosine_pearson_jaccard_diff'].mean()

print("코사인 유사도 + 피어슨 유사도 + 자카드 유사도: ", cosine_pearson_jaccard_diff_mean)

# 유클리디안 유사도 + 피어슨 유사도 + 자카드 유사도
for elem in sts_total:
    sts_total['euclidean_pearson_jaccard_similarity'] = sts_total.apply(lambda row: min(row['euclidean_similarity'], row['pearson_similarity'], row['jaccard_similarity']), axis=1)
sts_total['euclidean_pearson_jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['euclidean_pearson_jaccard_similarity'])
euclidean_pearson_jaccard_diff_mean = sts_total.iloc[1:]['euclidean_pearson_jaccard_diff'].mean()

print("유클리디안 유사도 + 피어슨 유사도 + 자카드 유사도: ", euclidean_pearson_jaccard_diff_mean)

print()

# 코사인 유사도 + 유클리디안 유사도 + 피어슨 유사도 + 자카드 유사도
for elem in sts_total:
    sts_total['cosine_euclidean_pearson_jaccard_similarity'] = sts_total.apply(lambda row: min(row['cosine_similarity'], row['euclidean_similarity'], row['pearson_similarity'], row['jaccard_similarity']), axis=1)
sts_total['cosine_euclidean_pearson_jaccard_diff'] = abs(sts_total['percent_score'] - sts_total['cosine_euclidean_pearson_jaccard_similarity'])
cosine_euclidean_pearson_jaccard_diff_mean = sts_total.iloc[1:]['cosine_euclidean_pearson_jaccard_diff'].mean()

print("코사인 유사도 + 유클리디안 유사도 + 피어슨 유사도 + 자카드 유사도: ", cosine_euclidean_pearson_jaccard_diff_mean)
