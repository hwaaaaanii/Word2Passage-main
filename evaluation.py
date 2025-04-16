import json, re, os
import argparse
from tqdm import tqdm 
from llm_module.prompt_templates import *
from llm_module.llama_model import *
import string
import openai
from openai import OpenAI

def llm_eval_gpt4o(query,ground_truths,prediction):
    
    openai.api_key = 'your_api_key'
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages = 
        [{"role": "system", "content": "You are an evaluation tool."},
        {"role": "user", "content": f"""
You are an evaluation tool . Just answer by {{ Yes }} or {{ No }}. 
Here is a question , a golden answer and an AI-generated answer. 
Judge whether the AI-generated answer is correct according to the question and golden answer ,
answer with {{ Yes }} or {{ No }}.

Question : {query}
Golden answer : {ground_truths}
Generated answer : {prediction}
Response : 
"""}],
        max_tokens=500,
        temperature=0.0,  
    )
    llm = response.choices[0].message.content
    
    return llm



def calculate_acc(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite Query")
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-method', type=str, nargs='+', required=True, help="Specify method for evaluation")
    parser.add_argument('-LLM', type=str, required=True)
    args = parser.parse_args()

    load_llama_model('Llama3.1_8b')

    total_result = {}
    total_output_path = f'./datasets/QA/{args.dataset}/{args.dataset}/result/{args.LLM}_metrics.json'

    output_dir = os.path.dirname(total_output_path)
    os.makedirs(output_dir, exist_ok=True)

    for option in args.method:
        acc_cnt, llm_cnt = 0, 0
        input_path = f'./datasets/QA/{args.dataset}/{args.dataset}/retrieved_docs/{args.LLM}_{option}_docs.json'
        output_path = f'./datasets/QA/{args.dataset}/{args.dataset}/retrieved_docs/{args.LLM}_{option}_results.json'

        with open(input_path, 'r', encoding='utf-8') as input_queries, open(output_path, 'w', encoding='utf-8') as output_file:
            lines = input_queries.readlines()
            for idx, line in enumerate(tqdm(lines, desc=f"Processing {option}", unit="query")):
                data = json.loads(line)
                query, gt_answers, retrieved_docs = data.get('query'), data.get('answers'), data.get('retrieved_docs')
                retrieved_text = '\n\n'.join(retrieved_docs)
                generation_prompt = answer_generation_prompt(query, retrieved_text)
                generated_answer = generate_answer(generation_prompt)

                match = re.search(r':\s*(.*?)}', generated_answer)
                generated_answer_parsed = match.group(1) if match else generated_answer.strip()

                llm_eval = llm_eval_gpt4o(query, gt_answers, generated_answer_parsed)
                print(llm_eval)
                if 'Yes' in llm_eval:
                    llm_cnt += 1

                acc = calculate_acc(generated_answer_parsed, gt_answers)
                acc_cnt += acc

                save_dic = {
                    'query': query,
                    'GT answer': gt_answers,
                    'Pred answer': generated_answer_parsed,
                    'ACC': acc,
                    'LLM Eval': llm_eval
                }

                output_file.write(json.dumps(save_dic) + '\n')  

        total_result[option] = {'ACC': acc_cnt / (idx + 1), 'LLM Eval': llm_cnt / (idx + 1)}
        print(f'Result of {option}: ACC - {acc_cnt / (idx + 1)}, LLM Eval - {llm_cnt / (idx + 1)}')

    with open(total_output_path, 'w', encoding='utf-8') as total_output_file:
        json.dump(total_result, total_output_file, indent=4, ensure_ascii=False)

    print(total_result)
