from openai import OpenAI
import similar_precedent
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

similarity_threshold = 65

userInput = "매매의 목적물이 화재로 소실됨으로써 매도인의 매매 목적물 인도 의무가 이행 불능일 경우 매수인이 화재사고로 매도인이 지급받게 되는 화재보험금 화재공제금 전부에 대하여 대상청구권을 행사할 수 있나요?"
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
                    'caseNumber': {similarData['사건번호']},
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
        model="gpt-4o",
        temperature=0
    )

    print(chat_completion.choices[0].message.content)
    
    
    