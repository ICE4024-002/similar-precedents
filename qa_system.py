from openai import OpenAI
import similar_precedent

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

userInput = "공무원의 불법행위로 손해를 입은 피해자의 국가배상청구권의 소멸시효 기간이 지났으나 국가가 소멸시효 완성을 주장하는 것이 신의성실의 원칙에 반하는 권리남용으로 허용될 수 없어 배상 책임을 이행한 경우에는 소멸시효 완성 주장이 권리남용에 해당하게 된 원인 행위와 관련하여 공무원이 원인이 되는 행위를 적극적으로 주도하였다는 등의 특별한 사정이 없는 한 국가가 공무원에게 구상권을 행사하는 것은 신의칙상 허용되지 않는 것인지요"
similarData = similar_precedent.get_similar_precedent(userInput)
print(f'>>> Similar precedent data: {similarData}')

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
            # "content" : f"""
            # Here is a info of relevant case law:
            #     'id': {similarData['판례정보일련번호']},
            #     'caseName': {similarData['사건명']},
            #     'caseNumber': {similarData['사건번호']},
            #     'sentenceDate': {similarData['선고일자']},
            #     'sentence': {similarData['선고']},
            #     'courtName': {similarData['법원명']},
            #     'judgementType': {similarData['판결유형']},
            #     'decision': {similarData['판시사항']},
            #     'judgementSummary': {similarData['판결요지']},
            #     'referenceArticles': {similarData['참조조문']},
            #     'referencePrecedents': {similarData['참조판례']},
            #     'fullText': {similarData['전문']}
            #     """,
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