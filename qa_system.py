from openai import OpenAI
import similar_precedent
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

similarity_threshold = 65

userInput = "술집에서 시비가 붙어 옆자리 사람과 싸움을 하게 되었습니다 합의를 하고 싶은데 어떻게 해야 하나요"
similarData, similarity = similar_precedent.get_similar_precedent(userInput)
print(f'>>> Similar precedent data: {similarData}')

prompt = {}

if similarity < similarity_threshold:
    prompt = {
        "role" : "system",
        "content" : "You are a Korean law expert tasked with providing clear and precise answers to various legal questions."
                    "Create a universal solution for user query and provide guidance in the last line of your answers to contact legal experts for details."
                    "Ensure your explanation is both comprehensive and accessible to non-expert users."
                    "You must answer in Korean."
    }
else:
    prompt = {
        "role" : "system",
        "content" : "You are a Korean law expert tasked with summarizing similar precedents and providing conclusions based on them."
                    "First, briefly summarize the relevant details from similar case law, focusing on judgementSummary and fullText."
                    "Conclude your response by explaining how these cases apply to the user's query, emphasizing the outcome of the precedents."
                    "Your responses should always end with a reference to specific articles or sections of the law that directly apply to the user's query, ensuring your advice is grounded in relevant legal principles."
                    "For each legal query, carefully analyze any given context or case law to extract pertinent legal precedents and principles."
                    "In responding to the user's query, consider both the general principles of law and any relevant case law or statutes that specifically address the issue at hand."
                    "Structure your response to start with a summary of the case, followed by a conclusion that outlines the legal basis for the advice, as follows: 'Based on the precedents, your situation can be concluded as follows... In accordance with Article [number] of [Law Name]'."
                    "Ensure your explanation is both comprehensive and accessible to non-expert users."
                    "You must answer in Korean."
    }

similar_data = {
                "role" : "system",
                "content" : f"""
                Here is a info of relevant case law:
                    'caseName': {similarData['사건명']},
                    'sentence': {similarData['선고']},
                    'caseNumber': {similarData['사건번호']},
                    'judgementType': {similarData['판결유형']},
                    'decision': {similarData['판시사항']},
                    'judgementSummary': {similarData['판결요지']},
                    'referenceArticles': {similarData['참조조문']},
                    'referencePrecedents': {similarData['참조판례']},
                    'fullText': {similarData['전문']}
                    """
}

userInput = {
                "role": "user",
                "content": f"{userInput}",
}

for i in range(1):
    messages_list = []
    if similarity < similarity_threshold:
        messages_list = [prompt, userInput]
    else:
        messages_list = [prompt, similar_data, userInput]

    print(">>> GPT generating...")
    chat_completion = client.chat.completions.create(
        messages=messages_list,
        model="gpt-3.5-turbo",
        temperature=0
    )

    print(chat_completion.choices[0].message.content)