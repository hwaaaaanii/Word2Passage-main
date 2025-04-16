import json, math
import argparse
from tqdm import tqdm
import pickle
import traceback
from collections import defaultdict, Counter
import ast, unicodedata, re, os
import pandas as pd
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
import random
from pyserini.search import LuceneSearcher, get_qrels
from config import IMPORTANCE_CONFIG, WORD_COUNT
noise_words = set(stopwords.words('english'))



# -----------------------------------------------------------------------------
# 1) Utility Functions for Query Processing
#    - get_importance
#    - remove_noise_words
#    - mapping_term_repetition
# -----------------------------------------------------------------------------
def get_importance(dataset: str, query_type: str):
    group1 = ['fiqa','trec-covid']
    group2 = ['dl19','dl20']
    group3 = ['scifact','scidocs','arguana']
    group4 = ['nfcorpus','webis-touche2020']
    if dataset in group1:
        shared_weight = 'fiqa'
    elif dataset in group2:
        shared_weight = 'msmarco'
    elif dataset in group3:
        shared_weight = 'scifact'
    elif dataset in group4:
        shared_weight = 'nfcorpus'
    else : 
        shared_weight = dataset
    if shared_weight in IMPORTANCE_CONFIG:
        if query_type in IMPORTANCE_CONFIG[shared_weight]:
            return IMPORTANCE_CONFIG[shared_weight][query_type]



def remove_noise_words(query, noise_words):
    words = query.split()
    filtered_words = [word for word in words if word not in noise_words]
    query = ' '.join(filtered_words)
    return query



def mapping_term_repetition(query: str) -> dict:
    dic = dict(Counter(query.split()))
    split_dic = {}
    for key, value in dic.items():
        words = key.split() 
        for word in words:
            if word not in split_dic:
                split_dic[word] = value
            else:
                split_dic[word] += value
    split_dic = dict(sorted(split_dic.items(), key=lambda item: item[1], reverse=True))
    return split_dic


# -----------------------------------------------------------------------------
# 2) Rewriting Functions
#    - Hyde_rewriter
#    - MuGI_rewriter
#    - W2P_rewriter
# -----------------------------------------------------------------------------
def Hyde_rewriter(query: str, passages: dict) -> str:
    HyDE = passages['HyDE']
    expanded_query = HyDE
    return expanded_query

    
    
def MuGI_rewriter(query: str, passages: dict, num: int) -> str:
    passage = passages['MuGI']
    passage = passage[:num]
    beta = 4
    total_ref = ' '.join(passage)
    factor = (len(total_ref) // len(query)) // beta  
    expanded_query = (query + ' ') * factor + total_ref
    return expanded_query



def word_reweight(query: str,
                  word: str,
                  sentence: str,
                  passage: str,
                  reference_dic: dict,
                  word_imp: float,
                  sentence_imp: float,
                  passage_imp: float) -> tuple:
    """
    Re-weight word importance in the reference based on the frequency
    in each of 'word', 'sentence', and 'passage', multiplied by their respective importance weights.
    """
    word_dic, sentence_dic, passage_dic = mapping_term_repetition(word), mapping_term_repetition(sentence), mapping_term_repetition(passage)
    total_words = set(word.split()) | set(sentence.split()) | set(passage.split())   
    for word in total_words:
        word_freq, sentence_freq, passage_freq = word_dic.get(word,0), sentence_dic.get(word,0), passage_dic.get(word,0)  
        word_factor, sentence_factor, passage_factor = word_imp*word_freq, sentence_imp*sentence_freq, passage_imp*passage_freq
        word_in_ref_imp = ((word_factor+sentence_factor+passage_factor))
        reference_dic[word] = round((word_in_ref_imp))  
    query = remove_noise_words(query, noise_words)
    for noise_word in noise_words:
        if noise_word in reference_dic:
            del reference_dic[noise_word]
    return query, reference_dic



def W2P_rewriter(query: str,
                 passages: list,
                 dataset: str,
                 domain_aware_factor: float,
                 num: int) -> str:
    """
    Rewrite the query using the 'Word2Passage' approach. For each pseudo-reference, we:
    1) Extract the 'word', 'sentence', 'passage'
    2) Re-weight them based on importance factors
    3) Aggregate them into a global dictionary
    4) Combine with the original query repeated a certain factor
    """
    pseudo_references_set = passages[-1]["W2P"]
    query_type = passages[0].get('query_type')
    word_imp, sentence_imp, passage_imp = get_importance(dataset, query_type)
    total_reference_dic ={}
    for idx, pseudo_reference in enumerate(pseudo_references_set):
        if idx < num: 
            word = ' '.join(map(str, pseudo_reference['words']))
            sentence = pseudo_reference['sentence']
            passage = pseudo_reference['passage']
            if isinstance(sentence, list):sentence = sentence[0]
            if isinstance(passage, list):passage = passage[0]
            
            concat_pseudo_reference = word + ' ' + sentence + ' ' + passage + ' '
            reference_dic = mapping_term_repetition(concat_pseudo_reference)
            query, reference_dic = word_reweight(query, word, sentence, passage, reference_dic, word_imp=word_imp, sentence_imp=sentence_imp, passage_imp=passage_imp)
            for key, value in reference_dic.items():
                total_reference_dic[key] = total_reference_dic.get(key, 0) + value
    query_dic = mapping_term_repetition(query)
    factor = round((sum(total_reference_dic.values())) / (sum(query_dic.values())))
    repeated_terms = ' '.join((key + ' ') * int(value * domain_aware_factor) for key, value in total_reference_dic.items())
    query = ' '.join((key + ' ') * value * factor for key, value in query_dic.items())

    expanded_query = query + repeated_terms
    return expanded_query



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rewrite Query")
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-num', type=int)
    parser.add_argument('-task', type=str, choices=['IR', 'QA'], required=True, help="Specify the task: IR or QA")
    parser.add_argument('-alpha', type=float, default=30)
    parser.add_argument('-LLM', type=str, required=True)
    args = parser.parse_args()
        
    datasets = args.dataset
    pseudo_path = f'./datasets/{args.task}/{args.dataset}/{args.dataset}/pseudo_references/{args.LLM}_{args.dataset}_pseudo_references.json'
    output_path = f'./datasets/{args.task}/{args.dataset}/{args.dataset}/expanded_query/{args.LLM}_{args.dataset}_expanded_queries.json'
                 
    with open(pseudo_path, 'r', encoding='utf-8') as reader:
        pseudo_lst = json.load(reader)

    pseudo = {}
    for idx, pseudo_dic in enumerate(pseudo_lst):
        query = list(pseudo_dic.keys())[0]
        pseudo_contents = pseudo_dic[query]
        pseudo[query] = pseudo_contents
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    qa_lst, rewritten_queries = [], []
    total_num = []

    if args.dataset == 'dl19' or args.dataset=='dl20' : dataset = 'msmarco'
    else : dataset = args.dataset
    
    avg_words_per_chunk = WORD_COUNT.get(dataset)
    domain_aware_factor = args.alpha/np.sqrt(avg_words_per_chunk)
    print(f'Domain Aware Factor : {domain_aware_factor}')

    for idx, query in enumerate(pseudo.keys()):
        pseudo_dic = pseudo[query]

        HyDE = Hyde_rewriter(query, pseudo_dic[-1])
        MuGI = MuGI_rewriter(query, pseudo_dic[-1], args.num)
        W2P = W2P_rewriter(query, pseudo_dic, args.dataset, domain_aware_factor = domain_aware_factor, num = args.num)
        dic = {'default':query, 'HyDE':HyDE, 'MuGI':MuGI, 'W2P':W2P}
        rewritten_queries.append(dic)
    
    
    with open(output_path, 'w') as output_file:
        json.dump(rewritten_queries, output_file, ensure_ascii=False, indent=4)

    print('='*100)
    print('Rewriting Done')

