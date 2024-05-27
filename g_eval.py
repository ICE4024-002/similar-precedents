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

def calculate_g_eval_score(question, answer):
    g_eval_template = f"""
You will be presented with one legal question and the system's answer to it. 
Your task is to evaluate the answers according to certain metrics. 
Your task is to rate the summary on three metrics: legal accuracy, relevance, and practicality. 
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

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
7. Provide rationales: Provide a rationale for your calculated scores for legal accuracy, relevance, and practicality.

Example:
1. User question: {question}
2. System's answer: {answer}
# Evaluation Form (scores, rationales):
- Legal accuracy: 
- Rationale for legal accuracy:
- Relevance: 
- Rationale for relevance:
- Practicality: 
- Rationale for practicality:
"""
    prompt_template = PromptTemplate(input_variables=["question", "answer"], template=g_eval_template)
    prompt = prompt_template.format(question=question, answer=answer)

    print(prompt)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt}
            ],
            model="gpt-3.5-turbo",
            temperature=0
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)