import pandas as pd
import os
from openai import OpenAI
import similar_precedent
from tqdm import tqdm

# Set up the OpenAI API client
os.environ["TOKENIZERS_PARALLELISM"] = "false"
client = OpenAI(api_key="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")

similarity_threshold = 70
output_data = []

# Load the CSV file
file_path = './total_qa_spell_checked.csv'
df = pd.read_csv(file_path)

# Define prompt templates
general_prompt = {
    "role": "system",
    "content": "You are a Korean law expert tasked with providing clear and precise answers to various legal questions."
               "Create a universal solution for user query and provide guidance in the last line of your answers to contact legal experts for details."
               "Ensure your explanation is both comprehensive and accessible to non-expert users."
               "You must answer in Korean."
}

specific_prompt = {
    "role": "system",
    "content": "You are a Korean law expert tasked with summarizing similar precedents and providing conclusions based on them."
               "First, briefly summarize the relevant details from similar case law, focusing on judgementSummary and fullText."
               "Conclude your response by explaining how these cases apply to the user's query, emphasizing the outcome of the precedents."
               "Your responses should always end with a reference to specific articles or sections of the law that directly apply to the user's query, ensuring your advice is grounded in relevant legal principles."
               "For each legal query, carefully analyze any given context or case law to extract pertinent legal precedents and principles."
               "In responding to the user's query, consider both the general principles of law and any relevant case law or statutes that specifically address the issue at hand."
               "Structure your response to start with a summary of the case, followed by a conclusion that outlines the legal basis for the advice, as follows: 'Based on the precedents, your situation can be concluded as follows... In accordance with Article [number] of [Law Name]'."
               "Ensure your explanation is both comprehensive and accessible to non-expert users."
               "You must answer in Korean."
}

# Process each question
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing questions"):
    user_question = row['question']
    human_answer = row['answer']
    
    # Get similar precedent data
    similarData, similarity = similar_precedent.get_similar_precedent(user_question)
    
    # Prepare the messages list for OpenAI API
    if similarity > similarity_threshold:
        similar_data = {
            "role": "system",
            "content": f"""
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
        messages_list = [specific_prompt, similar_data, {"role": "user", "content": user_question}]
    
        # Get the answer from OpenAI API
        chat_completion = client.chat.completions.create(
            messages=messages_list,
            model="gpt-4o",
            temperature=0
        )
        llm_answer = chat_completion.choices[0].message.content
        
        # Save the result if the similarity is above the threshold
        if similarity >= similarity_threshold:
            output_data.append([user_question, human_answer, llm_answer, similarity])
            print(len(output_data))
            
        
        # Stop if we have 50 entries
        if len(output_data) >= 50:
            break

# Create a DataFrame for the output data
output_df = pd.DataFrame(output_data, columns=['질문', '사람의 답변', 'LLM 답변', '유사도'])

# Save the DataFrame to a new CSV file
output_file_path = './50개질답.csv'
output_df.to_csv(output_file_path, index=False)

print("Process completed. The results have been saved to", output_file_path)
