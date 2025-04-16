import json
import os
import time
import re
import ast
import argparse
import multiprocessing
from multiprocessing import Pool, set_start_method
from functools import partial
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional, Union

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from nltk.corpus import stopwords

# Global constants
NOISE_WORDS = set(stopwords.words('english'))
QUERY_TYPES = ["description", "entity", "person", "numeric", "location"]
MAX_QUERIES_PER_TYPE = 100


# ===== Utility Functions =====
def get_avg_token_corpus(corpus_path: str) -> float:
    """Calculate average number of tokens per document in corpus."""
    total_tokens, lines = 0, 0
    with open(corpus_path, 'r', encoding='utf-8') as corpus:
        for lines, line in enumerate(corpus, 1):
            total_tokens += len(json.loads(line)['contents'].split())
    return total_tokens / lines if lines > 0 else 0


def parse_response(data_str: str) -> dict:
    """Parse LLM response into structured dictionary."""
    data_str = data_str.strip().replace('\n', ' ')
    data_dict = ast.literal_eval(data_str)
    return {
        'words': data_dict['word'],
        'sentence': data_dict['sentence'],
        'passage': data_dict['passage'],
    }


def ndcg_score(relevance_scores: List[int]) -> float:
    """Calculate Normalized Discounted Cumulative Gain for search results."""
    relevance_scores = np.array(relevance_scores)
    if relevance_scores.sum() == 0:
        return 0.0
    
    # Calculate DCG
    dcg = np.sum((2**relevance_scores - 1) / np.log2(np.arange(2, relevance_scores.size + 2)))
    
    # Calculate ideal DCG
    ideal_relevance = np.sort(relevance_scores)[::-1]
    idcg = np.sum((2**ideal_relevance - 1) / np.log2(np.arange(2, ideal_relevance.size + 2)))
    
    return dcg / idcg


def classify_query_type(query_text: str) -> str:
    """Classify query into one of predefined types using LLM."""
    qtype_prompt = query_type_classification_prompt(query_text)
    classification_result = generate(
        prompt=qtype_prompt,
        generate_func=generate_query_label
    )
    
    # Extract query type from response (e.g., "Query Type: person")
    matched = re.findall(r"Query Type: (\w+)", classification_result)
    return matched[0] if matched else "unknown"


def generate_pseudo_reference(query_text: str) -> dict:
    """Generate word-to-passage (W2P) pseudo references for a query."""
    w2p_prompt = W2P_generate_prompt(query_text)
    return generate_w2P(
        prompt=w2p_prompt,
        generate_func=generate_W2P,
        parse_output_func=parse_response
    )


def remove_noise_words(query: str, noise_words: Set[str]) -> str:
    """Remove stopwords from query."""
    words = query.split()
    filtered_words = [word for word in words if word not in noise_words]
    return ' '.join(filtered_words)


def mapping_term_repetition(query: str) -> Dict[str, int]:
    """Count term frequencies in query text."""
    word_counts = dict(Counter(query.split()))
    split_counts = {}
    
    for term, count in word_counts.items():
        words = term.split()
        for word in words:
            split_counts[word] = split_counts.get(word, 0) + count
            
    # Sort by frequency (descending)
    return dict(sorted(split_counts.items(), key=lambda item: item[1], reverse=True))


def word_reweight(
    query: str,
    word: str,
    sentence: str,
    passage: str,
    reference_dic: Dict[str, int],
    word_imp: float,
    sentence_imp: float,
    passage_imp: float
) -> Tuple[str, Dict[str, int]]:
    """Reweight terms based on their importance in different context levels."""
    # Get term frequencies from different contexts
    word_dic = mapping_term_repetition(word)
    sentence_dic = mapping_term_repetition(sentence)
    passage_dic = mapping_term_repetition(passage)
    
    # Combine all unique words
    total_words = set(word.split()) | set(sentence.split()) | set(passage.split())
    
    # Apply importance weights to each term
    for term in total_words:
        word_freq = word_dic.get(term, 0)
        sentence_freq = sentence_dic.get(term, 0)
        passage_freq = passage_dic.get(term, 0)
        
        # Apply importance factors
        word_factor = word_imp * word_freq
        sentence_factor = sentence_imp * sentence_freq
        passage_factor = passage_imp * passage_freq
        
        # Calculate combined importance
        word_in_ref_imp = word_factor + sentence_factor + passage_factor
        reference_dic[term] = round(word_in_ref_imp)
    
    # Clean up query and reference dictionary
    query = remove_noise_words(query, NOISE_WORDS)
    for noise_word in NOISE_WORDS:
        if noise_word in reference_dic:
            del reference_dic[noise_word]
            
    return query, reference_dic


def W2P_rewriter(
    query: str,
    passages: List[Dict],
    word_imp: float,
    sentence_imp: float,
    passage_imp: float,
    domain_aware_factor: float,
    num: int
) -> str:
    """Rewrite query using Word-to-Passage expansion technique."""
    pseudo_references_set = passages[-1]["W2P"]
    total_reference_dic = {}

    # Process up to 'num' pseudo references
    for idx, pseudo_reference in enumerate(pseudo_references_set):
        if idx >= num:
            break
            
        # Extract parts of the pseudo reference
        word = ' '.join(map(str, pseudo_reference['words']))
        sentence = pseudo_reference['sentence']
        passage = pseudo_reference['passage']
        
        # Handle list values
        if isinstance(sentence, list):
            sentence = sentence[0]
        if isinstance(passage, list):
            passage = passage[0]

        # Combine all parts
        concat_pseudo_reference = f"{word} {sentence} {passage} "
        reference_dic = mapping_term_repetition(concat_pseudo_reference)

        # Apply term importance weights
        query, reference_dic = word_reweight(
            query,
            word,
            sentence,
            passage,
            reference_dic,
            word_imp=word_imp,
            sentence_imp=sentence_imp,
            passage_imp=passage_imp
        )
        
        # Combine with total reference dictionary
        for key, value in reference_dic.items():
            total_reference_dic[key] = total_reference_dic.get(key, 0) + value

    # Calculate expansion factor
    query_dic = mapping_term_repetition(query)
    factor = round(sum(total_reference_dic.values()) / (sum(query_dic.values()) + 1e-8))
    
    # Build expanded query
    repeated_terms = ' '.join(
        (key + ' ') * int(value * domain_aware_factor)
        for key, value in total_reference_dic.items()
    )
    
    original_query_terms = ' '.join(
        (key + ' ') * (value * factor)
        for key, value in query_dic.items()
    )
    
    return f"{original_query_terms} {repeated_terms}"


def run_grid_search_for_query_type(args, corpus_index_path: str) -> Dict:
    """Run grid search to find optimal parameters for a specific query type."""
    (
        avg_words_per_chunk,
        query_type,
        results_for_this_type,
        query_dic,
        rewritten_path,
        word_importance_grid,
        sentence_importance_grid,
        passage_importance_grid,
        tqdm_position
    ) = args
    
    # Initialize searcher
    searcher = LuceneSearcher(corpus_index_path)

    # Calculate total parameter combinations
    total_combinations = (
        len(word_importance_grid) *
        len(sentence_importance_grid) *
        len(passage_importance_grid)
    )

    # Track best results
    best_ndcg = float('-inf')
    best_candidates = []
    grid_search_results = []
    start_time = time.time()

    # Progress bar
    with tqdm(
        total=total_combinations,
        desc=f"[{query_type}]",
        position=tqdm_position,
        leave=True
    ) as pbar:
        # Test all parameter combinations
        for (word_imp, sentence_imp, passage_imp) in itertools.product(
            word_importance_grid, sentence_importance_grid, passage_importance_grid
        ):
            # 1. Rewrite queries with current parameters
            rewritten_queries = []
            for result_dic in results_for_this_type:
                query = list(result_dic.keys())[0]
                pseudo_info = result_dic[query]
                
                new_query = W2P_rewriter(
                    query=query,
                    passages=pseudo_info,
                    word_imp=word_imp,
                    sentence_imp=sentence_imp,
                    passage_imp=passage_imp,
                    domain_aware_factor=30/np.sqrt(avg_words_per_chunk),
                    num=5
                )
                rewritten_queries.append({"default": query, "W2P": new_query})

            # 2. Evaluate search performance
            ndcg_scores = []
            for dic in rewritten_queries:
                default_query_text = dic['default']
                rewritten_text = dic['W2P']
                
                # Search with rewritten query
                hits = searcher.search(rewritten_text, k=10)
                
                # Get ground truth documents
                gt_docs = query_dic.get(default_query_text, [])
                
                # Calculate relevance
                relevance_list = []
                for hit in hits:
                    doc = searcher.doc(hit.docid)
                    doc_json = json.loads(doc.raw())
                    relevance_list.append(1 if doc_json['contents'] in gt_docs else 0)
                
                ndcg_scores.append(ndcg_score(relevance_list))

            # 3. Compute average nDCG
            average_ndcg = float(np.mean(ndcg_scores))
            candidate = (word_imp, sentence_imp, passage_imp)

            # Update best candidates
            if average_ndcg > best_ndcg:
                best_ndcg = average_ndcg
                best_candidates = [candidate]
            elif np.isclose(average_ndcg, best_ndcg):
                best_candidates.append(candidate)

            # Tie-breaking among current best
            current_best_candidate = select_best_candidate(best_candidates)

            # Store grid-search result
            grid_search_results.append({
                "word_imp": word_imp,
                "sentence_imp": sentence_imp,
                "passage_imp": passage_imp,
                "average_ndcg": average_ndcg
            })

            # Update progress bar
            pbar.set_postfix({
                "current_best": f"(Word={current_best_candidate[0]}, "
                                f"Sent={current_best_candidate[1]}, "
                                f"Pass={current_best_candidate[2]}, "
                                f"nDCG={best_ndcg:.4f})"
            })
            pbar.update(1)

    # Final tie-break
    final_best = select_best_candidate(best_candidates)
    elapsed = time.time() - start_time
    
    return {
        "query_type": query_type,
        "best_params": {
            "word_imp": final_best[0],
            "sentence_imp": final_best[1],
            "passage_imp": final_best[2]
        },
        "best_ndcg": best_ndcg,
        "grid_search_results": grid_search_results,
        "elapsed_time": elapsed
    }


def select_best_candidate(candidates: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Select best parameter combination using tie-breaking rules."""
    if not candidates:
        return (None, None, None)
        
    # Rule 1: Prefer higher word importance
    max_word = max(c[0] for c in candidates)
    candidates_word = [c for c in candidates if c[0] == max_word]
    
    # Rule 2: Prefer lower passage importance
    min_passage = min(c[2] for c in candidates_word)
    candidates_passage = [c for c in candidates_word if c[2] == min_passage]
    
    # Rule 3: Prefer median sentence importance
    sentence_vals = sorted(c[1] for c in candidates_passage)
    median_sentence = np.median(sentence_vals)
    
    # Find closest to median
    return min(
        candidates_passage,
        key=lambda c: abs(c[1] - median_sentence)
    )


def load_or_generate_pseudo_references(
    train_path: str,
    output_path: str,
    query_dic: Dict[str, List[str]]
) -> List[Dict]:
    """Load existing pseudo references or generate new ones."""
    if os.path.exists(output_path):
        print(f"Loading existing pseudo references from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as fr:
            results = json.load(fr)
        print(f"Loaded {len(results)} pseudo references")
        return results
    
    # Initialize LLM
    from llm_module.llama_model import load_llama_model
    load_llama_model('Llama3.1_8b')
    
    print(f"Generating new pseudo references")
    type_counts = {qtype: 0 for qtype in QUERY_TYPES}
    results = []
    
    # Load training data
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each query
    for line in lines:
        line_data = json.loads(line)
        q_text = line_data['query_text']
        gt_chunk = line_data['GT_chunk']
        query_dic[q_text] = gt_chunk
        
        # Classify query type
        qtype = classify_query_type(q_text)
        
        # Check if we need more of this type
        if qtype in type_counts and type_counts[qtype] < MAX_QUERIES_PER_TYPE:
            # Generate pseudo-reference
            gen_w2p_output = generate_pseudo_reference(q_text)
            
            # Build reference structure
            reference_dic = {
                q_text: [
                    {'query_type': [qtype]},
                    {'W2P': gen_w2p_output}
                ]
            }
            results.append(reference_dic)
            
            # Update progress
            type_counts[qtype] += 1
            print(f"[{qtype}] => {type_counts[qtype]}/{MAX_QUERIES_PER_TYPE} now generated.")
            
            # Check if all types have reached the limit
            if all(count >= MAX_QUERIES_PER_TYPE for qtype, count in type_counts.items() 
                   if qtype in QUERY_TYPES):
                print(f"We have {MAX_QUERIES_PER_TYPE} queries for each of the query types.")
                break
    
    # Save results to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fw:
        json.dump(results, fw, ensure_ascii=False, indent=4)
        
    print(f"Saved pseudo references to {output_path}")
    print(f"Final type counts: {type_counts}")
    
    return results


def main():
    """Main execution function."""
    # Initialize multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate pseudo passages and optimize query expansion")
    parser.add_argument('-LLM', type=str, required=True, help="LLM model name")
    parser.add_argument('-corpus_name', type=str, required=True, help="Corpus name")
    args = parser.parse_args()

    # Define paths
    train_path = './datasets/user_data/train_subsample_processed.json'
    rewritten_path = './datasets/user_data/temp.json'
    output_path = f'./datasets/user_data/pseudo_references/{args.LLM}_pseudo_references.json'
    corpus_index_path = f'./datasets/user_data/lucene_index/{args.corpus_name}'
    corpus_path = f'./datasets/user_data/corpus/{args.corpus_name}.jsonl'
    
    # Get average words per chunk in corpus
    avg_words_per_chunk = get_avg_token_corpus(corpus_path)
    
    # Initialize query dictionary
    query_dic = {}
    
    # Load or generate pseudo references
    results = load_or_generate_pseudo_references(train_path, output_path, query_dic)
    
    # Load ground truth data
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q_text = data['query_text']
            gt_chunk = data['GT_chunk']
            query_dic[q_text] = gt_chunk

    # Group pseudo references by query_type
    results_by_type = defaultdict(list)
    for rd in results:
        query_text = list(rd.keys())[0]
        pseudo_info = rd[query_text]
        q_type_list = pseudo_info[0].get('query_type', [])
        q_type = q_type_list if q_type_list else 'unknown'
        results_by_type[q_type].append(rd)

    # Define the parameter grids
    word_importance_grid = [round(i * 0.2, 1) for i in range(1, 11)]  # 0.2 to 2.0
    sentence_importance_grid = [round(i * 0.2, 1) for i in range(1, 11)]  # 0.2 to 2.0
    passage_importance_grid = [round(i * 0.2, 1) for i in range(1, 11)]  # 0.2 to 2.0

    # Build argument tuples for each query_type
    query_type_list = list(results_by_type.keys())
    pool_args = []
    
    for i, qt in enumerate(query_type_list):
        arg_tuple = (
            avg_words_per_chunk,
            qt,
            results_by_type[qt],
            query_dic,
            rewritten_path,
            word_importance_grid,
            sentence_importance_grid,
            passage_importance_grid,
            i  # Position in tqdm
        )
        pool_args.append(arg_tuple)
        
    # Run the grid search in parallel
    partial_func = partial(run_grid_search_for_query_type, corpus_index_path=corpus_index_path)
    with Pool(processes=min(len(query_type_list), multiprocessing.cpu_count())) as pool:
        results_pool = pool.map(partial_func, pool_args)

    # Process results
    final_summary = {}
    os.makedirs('./datasets/user_data/grid_search/', exist_ok=True)
    
    for res in results_pool:
        qt = res["query_type"]
        final_summary[qt] = {
            "best_params": res["best_params"],
            "best_ndcg": res["best_ndcg"],
            "elapsed_time": res["elapsed_time"]
        }
        
        # Save detailed results
        detail_path = f'./datasets/user_data/grid_search/grid_search_results_{qt}.json'
        with open(detail_path, 'w', encoding='utf-8') as fw:
            json.dump(res["grid_search_results"], fw, ensure_ascii=False, indent=4)
        print(f"[{qt}] Detailed grid search results saved to {detail_path}")
        
    # Save summary of best parameters
    summary_path = './datasets/user_data/grid_search/summary_best_params_by_type.json'
    with open(summary_path, 'w', encoding='utf-8') as fw:
        json.dump(final_summary, fw, ensure_ascii=False, indent=4)
        
    # Print final summary
    print("\n=== Grid Search Completed ===")
    for qt, info in final_summary.items():
        bp = info["best_params"]
        ndcg_val = info["best_ndcg"]
        etime = info["elapsed_time"]
        print(f"Query Type: {qt} | Best Params={bp}, Best nDCG={ndcg_val:.4f}, Elapsed={etime:.2f}s")

    print(f"\nFinal summary saved to: {summary_path}")
    
    # Clean up multiprocessing resources
    import gc
    gc.collect()
    for p in multiprocessing.active_children():
        p.join(timeout=1)
    print("\nAll multiprocessing resources cleaned up. Program terminated cleanly.")


if __name__ == '__main__':
    main()