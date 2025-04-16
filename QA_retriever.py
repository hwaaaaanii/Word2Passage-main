import json
import argparse
import os
from tqdm import tqdm
from pyserini.search import LuceneSearcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beir Evaluation with pytrec-eval")
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-method', type=str, nargs='+', required=True, help="Specify method for evaluation")
    parser.add_argument('-LLM', type=str, required=True)
    args = parser.parse_args()
    for option in args.method:
        answer_path = f'./datasets/QA/{args.dataset}/{args.dataset}/test_subsample_processed.json'
        rewritten_path = f'./datasets/QA/{args.dataset}/{args.dataset}/expanded_query/{args.LLM}_{args.dataset}_expanded_queries.json'
        output_path = f'./datasets/QA/{args.dataset}/{args.dataset}/retrieved_docs/{args.LLM}_{option}_docs.json'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        answer_dic, queries = {}, {}

        with open(answer_path, 'r', encoding='utf-8') as answers:
            for line in answers:
                data = json.loads(line)
                answer_dic[data.get('query_text')] = data.get('answers')

        if args.dataset in ['nq', 'hotpotqa', 'fiqa']:
            searcher = LuceneSearcher.from_prebuilt_index(f'beir-v1.0.0-{args.dataset}.flat')
            doc_name = 'text'
        else:
            searcher = LuceneSearcher('./datasets/data/lucene_index/psgs_w100')
            doc_name = 'contents'
            
        with open(rewritten_path, 'r', encoding='utf-8') as rewritten_lst:
            data = json.load(rewritten_lst)

        for query_dic in data:
            default = query_dic.get('default')
            expanded = query_dic.get(option)
            queries[expanded] = default
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for query in tqdm(queries.keys(), desc=f"Retrieving Option : {option}", unit="query"):
                default_query = queries.get(query)
                try : 
                    hits = searcher.search(query, k=10)
                except:
                    hits = searcher.search('default_query', k=10)
                doc_lst = []
                
                for hit in hits:
                    doc = searcher.doc(hit.docid)
                    doc_dic = json.loads(doc.raw())
                    doc_lst.append(doc_dic[doc_name])
                
                answers = answer_dic.get(default_query, [])
                save_dic = {'query': default_query, 'answers': answers, 'retrieved_docs': doc_lst}
                
                f.write(json.dumps(save_dic, ensure_ascii=False) + '\n')
