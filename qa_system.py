from openai import OpenAI
import os
from langchain.prompts import PromptTemplate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

open_api_key = os.getenv('OPEN_API_KEY')
client = OpenAI(
    # This is the default and can be omitted
    api_key=open_api_key,
)

similarity_threshold = 65

# 유사도가 낮은 경우의 프롬프트 템플릿
low_similarity_template = """
You are a Korean law expert tasked with providing clear and precise answers to various legal questions.
Create a universal solution for user query and provide guidance in the last line of your answers to contact legal experts for details.
Ensure your explanation is both comprehensive and accessible to non-expert users.
You must answer in Korean.
"""

# 유사도가 높은 경우의 프롬프트 템플릿
high_similarity_template = """
You are a Korean law expert tasked with summarizing similar precedents and providing conclusions based on them.
First, briefly summarize the relevant details from similar case law, focusing on judgementSummary and fullText.
Conclude your response by explaining how these cases apply to the user's query, emphasizing the outcome of the precedents.
Your responses should always end with a reference to specific articles or sections of the law that directly apply to the user's query, ensuring your advice is grounded in relevant legal principles.
For each legal query, carefully analyze any given context or case law to extract pertinent legal precedents and principles.
In responding to the user's query, consider both the general principles of law and any relevant case law or statutes that specifically address the issue at hand.
Structure your response to start with a summary of the case, followed by a conclusion that outlines the legal basis for the advice, as follows: 'Based on the precedents, your situation can be concluded as follows... In accordance with Article [number] of [Law Name]'.
Ensure your explanation is both comprehensive and accessible to non-expert users.
You must answer in Korean.
"""

low_similarity_prompt = PromptTemplate(input_variables=[], template=low_similarity_template)
high_similarity_prompt = PromptTemplate(input_variables=["similarPrecedent", "expertEvaluation"], template=high_similarity_template)

def get_gpt_answer_by_precedent(question, similar_precedent, similarity):
    if similarity < similarity_threshold:
        prompt_content = low_similarity_prompt.format()
    else:
        similar_precedent_content = f"""
        Here is a info of relevant case law:
            'caseName': {similar_precedent['사건명']},
            'sentence': {similar_precedent['선고']},
            'caseNumber': {similar_precedent['사건번호']},
            'judgementType': {similar_precedent['판결유형']},
            'decision': {similar_precedent['판시사항']},
            'judgementSummary': {similar_precedent['판결요지']},
            'referenceArticles': {similar_precedent['참조조문']},
            'referencePrecedents': {similar_precedent['참조판례']},
            'fullText': {similar_precedent['전문']}
        """

    # 비슷한 판례와 전문가 평가를 추가
    prompt_content = high_similarity_prompt.format(similarPrecedent=similar_precedent_content, expertEvaluation="")

    user_input = {
        "role": "user",
        "content": f"{question}",
    }

    try:
        messages_list = [
            {"role": "system", "content": prompt_content},
            user_input
        ]

        chat_completion = client.chat.completions.create(
            messages=messages_list,
            model="gpt-3.5-turbo",
            temperature=0
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)