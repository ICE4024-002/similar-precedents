from openai import OpenAI
import similar_precedent

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

userInput = "저는 甲대학병원에서 전공의 과정을 밟고 있는 레지던트입니다. 그런데 몇 달 전부터 저에 대한 임금이 지급되지 않고 있어 병원측에 그 지급을 요구하였으나, 병원측은 레지던트 과정도 교육의 과정이므로 임금을 지급하지 않아도 되며 그 전에 얼마씩 지급한 것도 장학금 내지 생활비조로 병원측에서 호의적으로 지급한 것이었다고 말하고 있습니다. 저는 계속 이를 다투었다가는 장래에 좋지 않은 영향을 줄 것도 같아 어떻게 하여야 할지 모르겠습니다. 좋은 방법이 있는지요?"
sampleData = similar_precedent.get_similar_precedent(userInput)
print(sampleData)

print(">>> GPT generating...")
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role" : "system",
            "content" : "You are a Korean law expert tasked with providing clear and precise answers to various legal questions." 
                        "Your responses should always reference specific articles or sections of the law that directly apply to the user's query, ensuring your advice is grounded in relevant legal principles."
                        "For each legal query, carefully analyze any given context or case law to extract pertinent legal precedents and principles."
                        "In responding to the user's query, consider both the general principles of law and any relevant case law or statutes that specifically address the issue at hand."
                        "Your response should be structured as follows: 'In accordance with Article [number] of [Law Name], your situation is addressed as follows...'."
                        "Ensure your explanation is both comprehensive and accessible to non-expert users."
                        "You must answer in Korean."
                        
        },
        {
            "role" : "system",
            "content" : f"""
            Here is a info of relevant case law:
                'id': {sampleData['판례정보일련번호']},
                'caseName': {sampleData['사건명']},
                'caseNumber': {sampleData['사건번호']},
                'sentenceDate': {sampleData['선고일자']},
                'sentence': {sampleData['선고']},
                'courtName': {sampleData['법원명']},
                'judgementType': {sampleData['판결유형']},
                'decision': {sampleData['판시사항']},
                'judgementSummary': {sampleData['판결요지']},
                'referenceArticles': {sampleData['참조조문']},
                'referencePrecedents': {sampleData['참조판례']},
                'fullText': {sampleData['전문']}
                """
        },
        {
            "role": "user",
            "content": f"{userInput}",
        }
    ],
    model="gpt-3.5-turbo",
    temperature=0
)

print(chat_completion.choices[0].message.content)