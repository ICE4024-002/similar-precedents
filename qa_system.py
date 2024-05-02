from openai import OpenAI
import similar_precedent
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

similarity_threshold = 65

userInput = "형님 부부가 사망하여 10살인 조카의 미성년후견인이 되었습니다 불쌍한 조카를 제 양자로 입양하고 싶은데 어떻게 해야 하나요"
similarData, similarity = similar_precedent.get_similar_precedent(userInput)
print(f'>>> Similar precedent data: {similarData}')

prompt = {
    "role" : "system",
    "content" : "You are a Korean law expert tasked with providing clear and precise answers to various legal questions." 
                "Your responses should always reference specific articles or sections of the law that directly apply to the user's query, ensuring your advice is grounded in relevant legal principles."
                "For each legal query, carefully analyze any given context or case law to extract pertinent legal precedents and principles."
                "In responding to the user's query, consider both the general principles of law and any relevant case law or statutes that specifically address the issue at hand."
                "Your response should be structured as follows: 'In accordance with Article [number] of [Law Name], your situation is addressed as follows...'."
                "Ensure your explanation is both comprehensive and accessible to non-expert users."
                "You must answer in Korean."                         
}

similar_data = {
                "role" : "system",
                "content" : f"""
                Here is a info of relevant case law:
                    'caseName': {similarData['사건명']},
                    'sentence': {similarData['선고']},
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