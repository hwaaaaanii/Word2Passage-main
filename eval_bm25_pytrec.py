import json
import argparse
import warnings
from tqdm import tqdm
import pytrec_eval
from pyserini.search import LuceneSearcher, get_qrels
from collections import defaultdict, Counter
import pandas as pd
import os

warnings.filterwarnings("ignore", category=FutureWarning)

def load_queries(query_path):
    queries = {}
    with open(query_path, 'r', encoding='utf-8') as reader:
        for idx, line in enumerate(reader):
            data = json.loads(line)
            queries[str(data["query_text"])] = str(data.get("qid", idx))

    return queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beir Evaluation with pytrec-eval")
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-task', type=str, choices=['IR', 'QA'], required=True, help="Specify the task: IR or QA")
    parser.add_argument('-method', type=str, nargs='+', required=True, help='method for rewritten queries')
    parser.add_argument('-LLM', type=str, required=True)
    args = parser.parse_args()

    expanded_query_path = f'./datasets/{args.task}/{args.dataset}/{args.dataset}/expanded_query/{args.LLM}_{args.dataset}_expanded_queries.json'
    query_path = f'./datasets/{args.task}/{args.dataset}/{args.dataset}/test_subsample_processed.json'
    
    if args.dataset == 'dl19' or args.dataset == 'dl20':
        searcher = LuceneSearcher.from_prebuilt_index(f'msmarco-v1-passage')
        qrels = get_qrels(f'{args.dataset}-passage')
    else: 
        searcher = LuceneSearcher.from_prebuilt_index(f'beir-v1.0.0-{args.dataset}.flat')
        qrels = get_qrels(f'beir-v1.0.0-{args.dataset}-test')
        
    qrels = {
        str(qid): {str(docid): int(rel) for docid, rel in rel_dict.items()}
        for qid, rel_dict in qrels.items()
    }

    queries = load_queries(query_path)


    with open(expanded_query_path, 'r', encoding='utf-8') as reader:
        rewritten_data = json.load(reader)

    total_result = {}
    metric = {'ndcg_cut_10'}

    for option in tqdm(args.method, desc="Evaluating method"):
        run_all = {}
        qrels_used = {}
        print(f'Total processed # of queries : {len(rewritten_data)}')
        processed_query_cnt = 0
        for dic in rewritten_data:
            default_query = dic['default']
            query_text = dic[option]
            query_id = queries.get(default_query, None)
            if query_id is None:
                continue
            qrel_dict = qrels.get(query_id, {})
            if not qrel_dict:
                print(f"Warning: No qrels found for query_id: {query_id}")
                continue
            processed_query_cnt+=1
            try:
                hits = searcher.search(query_text, k=10)
                run_dict = {str(hit.docid): hit.score for hit in hits}
            except:
                print('Maxclause problem, retreive with default query')
                hits = searcher.search(default_query, k=10)
                run_dict = {str(hit.docid): hit.score for hit in hits}
                
            run_all[query_id] = run_dict
            qrels_used[query_id] = qrel_dict

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_used, metric)
        results = evaluator.evaluate(run_all)
        
        ndcg_scores = [results[qid].get("ndcg_cut_10", 0.0) for qid in results]
        average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        total_result[option] = average_ndcg
        print(f'Processed query in Test : {processed_query_cnt}')
    print(f'Avg nDCG@10 of Dataset "{args.dataset}" : {total_result}')
