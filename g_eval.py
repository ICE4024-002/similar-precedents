import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate

dotenv_path = os.path.join(os.path.dirname(__file__), './.env.local')
load_dotenv(dotenv_path)

open_api_key = os.getenv('OPEN_API_KEY')
client = OpenAI(
    # This is the default and can be omitted
    api_key=open_api_key,
)

# 점수가 3점 미만인 항목을 반환
def evaluate_scores(scores):
    below_threshold = {}
    
    for category, score in scores.items():
        if score < 3:
            below_threshold[category] = score
            
    return below_threshold

def calculate_g_eval_score(question, answer):
    g_eval_template = f"""
                        You will be presented with one legal question and the system's answer to it.
                        Your task is to evaluate the answers according to certain metrics.
                        Your task is to rate the summary on three metrics: legal accuracy, relevance, and practicality.
                        Please make sure you read and understand these instructions carefully.
                        Please keep this document open while reviewing, and refer to it as needed.
                        
                        Evaluation Criteria:
                        1. Legal accuracy (1-5) - The extent to which the answer is consistent with the legal basis. Answers should accurately reflect relevant law and case law. Answers that are uncertain about the legal basis or contain incorrect information should be penalized.
                        2. Relevance (1-5) - The degree to which the answer directly addresses the user's question. The answer should be comprehensive and directly respond to the legal question posed. Answers that are off-topic, incomplete, or fail to address the main points of the question should be penalized.
                        3. Practicality (1-5) - The extent to which the answer provides practical, actionable advice. The answer should be useful and offer clear guidance or next steps. Answers that are vague, overly theoretical, or impractical should be penalized.
                        
                        Evaluation Steps:
                        1. Read the questions and answers: Carefully read the legal question provided and the system's answer.
                        2. Compare legal basis: Evaluate how well the answer responds to the question and how accurately it uses relevant laws and case law.
                        3. Rate legal accuracy: Evaluate how legally accurate the answer is and whether it contains false or unnecessary information.
                        4. Assess relevance: Determine how well the answer addresses the user's question and whether it provides a comprehensive response.
                        5. Evaluate practicality: Assess how practical and actionable the answer is, considering whether it provides useful guidance or clear next steps.
                        6. Assign scores: Score the answer for legal accuracy, relevance, and practicality on a scale of 1 to 5. Where 1 is highly inaccurate/irrelevant/impractical and 5 is perfectly accurate/relevant/practical.
                        7. In the end, you should only output items and scores. Do not include any other information in the output.

                        1. User question: {question}
                        2. System's answer: {answer}

                        # Evaluation Form
                        - Legal accuracy: (score)
                        - Relevance: (score)
                        - Practicality: (score)
                        """
    prompt_template = PromptTemplate(input_variables=["question", "answer"], template=g_eval_template)
    prompt = prompt_template.format(question=question, answer=answer)

    print(prompt)
    

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt}
            ],
            model="gpt-4o",
            temperature=0
        )
        
        evaluation = chat_completion.choices[0].message.content
        scores = {}
        # 항목 및 점수 파싱해서 scores에 저장
        # 저장형식 {'Legal accuracy': 5, 'Relevance': 5, 'Practicality': 5}
        for line in evaluation.split("\n"):
            if "Legal accuracy" in line:
                scores["Legal accuracy"] = int(line.split(": ")[1])
            elif "Relevance" in line:
                scores["Relevance"] = int(line.split(": ")[1])
            elif "Practicality" in line:
                scores["Practicality"] = int(line.split(": ")[1])
                
        below_threshold = evaluate_scores(scores)
        
        if below_threshold:
            # 3점 미만인 항목이 있을 경우 코드 작성
            pass

        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)