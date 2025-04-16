from query_rewriter import *
from pseudo_generator import *
from pyserini.search import LuceneSearcher
import argparse
import time
import json

def get_importance(grid_path: str, query_type: str):
    try:
        with open(grid_path, 'r', encoding='utf-8') as reader:
            data = json.load(reader)
            importances = data.get(query_type[0])
            word_imp =importances["best_params"]["word_imp"]
            sentence_imp =importances["best_params"]["sentence_imp"]
            passage_imp =importances["best_params"]["passage_imp"]
        return word_imp, sentence_imp, passage_imp
    except:
        return 1,1,1
        


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
                 query_type: str,
                 passages: list,
                 domain_aware_factor: float,
                 num: int,
                 grid_path) -> str:
    """
    Rewrite the query using the 'Word2Passage' approach. For each pseudo-reference, we:
    1) Extract the 'word', 'sentence', 'passage'
    2) Re-weight them based on importance factors
    3) Aggregate them into a global dictionary
    4) Combine with the original query repeated a certain factor
    """
    pseudo_references_set = passages
    word_imp, sentence_imp, passage_imp = get_importance(grid_path, query_type)
    print(word_imp, sentence_imp, passage_imp )
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


# -----------------------------------------------------------------------------
# 1) Main 
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo passages")
    parser.add_argument('-LLM', type=str, required=True)
    parser.add_argument('-corpus_name', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', help='Print retrieved contexts if True')
    args = parser.parse_args()

    if args.LLM.startswith('Qwen2.5'):
        from llm_module.qwen_model import *
        set_seed()
        load_qwen_model(args.LLM) 
    elif args.LLM.startswith('Llama3.1'):
        from llm_module.llama_model import *
        set_seed()
        load_llama_model(args.LLM)
    else:
        raise ValueError(f"Unsupported LLM option: {args.LLM}")

    corpus_name = args.corpus_name

    # Initialize Lucene searcher with specified index
    searcher = LuceneSearcher(f'./datasets/user_data/lucene_index/{corpus_name}')

    # -----------------------------------------------------------------------------
    # 2) Model loading
    #    - Loads the specified LLM model for query processing and generation tasks
    # -----------------------------------------------------------------------------
    # Parameters for domain-aware query rewriting
    avg_words_per_chunk = 10
    domain_aware_factor = 30 / np.sqrt(avg_words_per_chunk)

    while True:
        query = input("Enter your query (or 'exit' to stop): " )
        if query.lower() == 'exit':
            print("Exiting...")
            break

        # -----------------------------------------------------------------------------
        # 3) Prompt generation for rewriting and classification
        #    - Generates prompts for query rewriting (W2P) and query type classification
        # -----------------------------------------------------------------------------
        W2P_prompt = W2P_generate_prompt(query)
        query_type_prompt = query_type_classification_prompt(query)

        # -----------------------------------------------------------------------------
        # 4) Query type classification and rewriting
        #    - Classifies query type and rewrites the original query accordingly
        # -----------------------------------------------------------------------------
        grid_path = './datasets/user_data/grid_search/summary_best_params_by_type.json'
        start_time = time.time()
        generated_query_type = generate(prompt=query_type_prompt, generate_func=generate_query_label)
        generated = generate_w2P(prompt=W2P_prompt, generate_func=generate_W2P, parse_output_func=parse_response, num_of_generation=3)
        query_type= re.findall(r"Query Type: (\w+)", generated_query_type)
        expanded_query = W2P_rewriter(query, query_type, generated, domain_aware_factor, 3, grid_path)
        end_time = time.time()
        reference_generation_time = end_time - start_time

        print(f"\nReference generation time: {reference_generation_time:.2f} sec")

        # -----------------------------------------------------------------------------
        # 5) Document retrieval from Lucene
        #    - Performs retrieval of top-k relevant documents using expanded query
        # -----------------------------------------------------------------------------
        start_time = time.time()
        hits = searcher.search(expanded_query, k=10)
        retrieved_text = []
        for hit in hits:
            doc = searcher.doc(hit.docid)
            doc_dic = json.loads(doc.raw())
            retrieved_text.append(doc_dic['contents'])
        end_time = time.time()
        retrieval_time = end_time - start_time

        print(f"Retrieving time: {retrieval_time:.2f} sec")

        # -----------------------------------------------------------------------------
        # 6) Answer generation
        #    - Generates a final answer based on retrieved documents and original query
        # -----------------------------------------------------------------------------
        generation_prompt = answer_generation_prompt(query, retrieved_text)
        generated_answer = generate_answer_do_sample(generation_prompt)


        # Output total pipeline execution time and the generated answer
        print(f"Full pipeline time: {reference_generation_time + retrieval_time:.2f} sec\n")
        print('='*50)
        if args.verbose:
            print(f'Retrieved contexts : ')
            for idx, context in enumerate(retrieved_text):
                print(f'{idx+1} : {context}')
            print('-'*50)
        print(f'Query : {query}, type : {query_type} \nLLM Response : {generated_answer[12:-2]}')
        print('='*50)   
