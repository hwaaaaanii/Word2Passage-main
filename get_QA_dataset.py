import os, re, unicodedata
import subprocess
import random 
from datasets import load_dataset
import json, shutil
import pandas as pd
from collections import defaultdict
from beir.beir import util, LoggingHandler

# -----------------------------------------------------------------------------
# 1) Basic setup: Random seed & utility function for text normalization
# -----------------------------------------------------------------------------
random.seed(13370)  # Don't change.



def normalize_text(text: str) -> str:
    """
    Convert to lowercase, remove unwanted characters, and collapse whitespace.
    """
    text = text.strip()
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^가-힣a-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text



def move_contents(source, destination):
    """
    Moves all files and subfolders from 'source' into 'destination'.
    If 'destination' does not exist, it will be created.
    """
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    for item in os.listdir(source):
        # Skip the destination if it's inside source (avoid recursion issues)
        if os.path.normpath(os.path.join(source, item)) == os.path.normpath(destination):
            continue

        origin_path = os.path.join(source, item)
        shutil.move(origin_path, destination)



# -----------------------------------------------------------------------------
# 2) get_nq_dataset:
#    - Download & unzip BEIR dataset
#    - Merge with google's NQ Open
#    - Write processed dataset (question, answers, GT_chunk)
#    - Sub-sample for final test set
# -----------------------------------------------------------------------------
def get_nq_dataset(dataset):
    print('-'*50)
    print(f'Processing dataset : {dataset}')
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = f"./datasets/QA"
    data_path = util.download_and_unzip(url, out_dir)    
    old_dir = os.path.join(out_dir, dataset)           
    nested_dir = os.path.join(out_dir, dataset, dataset)
    
    if not os.path.exists(nested_dir):
        os.makedirs(nested_dir, exist_ok=True)
        for fname in os.listdir(old_dir):
            if fname == dataset:
                continue
            src_path = os.path.join(old_dir, fname)
            dst_path = os.path.join(nested_dir, fname)
            shutil.move(src_path, dst_path)    
            
    query_path = f"./datasets/QA/{dataset}/{dataset}/queries.jsonl"
    corpus_path = f"./datasets/QA/{dataset}/{dataset}/corpus.jsonl"
    qrel_path = f"./datasets/QA/{dataset}/{dataset}/qrels/test.tsv"      
    output_path = f'./datasets/QA/{dataset}/{dataset}/test_subsample_processed.json'
    nq_open = load_dataset('google-research-datasets/nq_open', split='validation')
    nq_open_qa_dict = {item['question']: item['answer'] for item in nq_open}
    
    query_dic, corpus_dic, save_dic = {}, {}, {}
    
    with open(query_path, 'r', encoding='utf-8') as queries:
        for line in queries:
            temp_dic = json.loads(line)
            if temp_dic['text'] in nq_open_qa_dict.keys():
                query_dic[temp_dic['_id']] = temp_dic['text']

    with open(corpus_path, 'r', encoding='utf-8') as corpus:
        for line in corpus:
            temp_dic = json.loads(line)
            corpus_dic[temp_dic['_id']] = temp_dic['text']
    
    qrel = pd.read_csv(qrel_path, sep='\t')
    query_doc_map = defaultdict(list)
    for _, row in qrel.iterrows():
        query_doc_map[row['query-id']].append(row['corpus-id'])
        
    with open(output_path, 'a', encoding='utf-8') as out_file:
        for query_id, doc_ids in query_doc_map.items():
            try:
                gt_chunks = [corpus_dic.get(doc_id) for doc_id in doc_ids]
                save_dic = {
                    'qid': query_id,
                    'query_text': query_dic.get(query_id),
                    'answers': nq_open_qa_dict.get(query_dic.get(query_id)),
                    'GT_chunk': gt_chunks
                }
                if save_dic['query_text'] is not None:
                    out_file.write(json.dumps(save_dic, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error processing query_id {query_id}: {e}")

    nq_zip_path = "./datasets/QA/nq.zip"
    if os.path.exists(nq_zip_path):
        os.remove(nq_zip_path)
        print(f"Deleted {nq_zip_path}")
    print('-'*50)
    sub_sample_dataset(dataset)



# -----------------------------------------------------------------------------
# 3) get_single_hop_dataset:
#    - Download DPR data (dev/train) for single-hop QA
#    - Process & sub-sample for final usage
# -----------------------------------------------------------------------------
def get_single_hop_dataset(dataset):
    print('-' * 50)
    print(f'Processing dataset : {dataset}')
    
    original_dir = os.getcwd()
    os.makedirs(f'./datasets/QA/{dataset}/{dataset}', exist_ok=True)
    os.chdir(f'./datasets/QA/{dataset}/{dataset}')
    
    if dataset == 'squad':
        files = [
            f'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-{dataset}1-dev.json.gz',
            f'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-{dataset}1-train.json.gz'
        ]
    else:
        files = [
            f'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-{dataset}-dev.json.gz',
            f'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-{dataset}-train.json.gz'
        ]
    
    for file_url in files:
        file_name = file_url.split('/')[-1]
        subprocess.run(['wget', file_url], check=True)
        if file_name.endswith('.gz'):
            subprocess.run(['gzip', '-d', file_name], check=True)

    os.chdir(original_dir)

    subprocess.run(['python', f'./processing_scripts/process_{dataset}.py'], check=True)
    subprocess.run([
        'python',
        './processing_scripts/subsample_dataset_and_remap_paras.py',
        dataset,
        'test',
        str(500)
    ], check=True)

    test_subsample_path = f'./datasets/QA/{dataset}/{dataset}/test_subsampled.jsonl'
    test_subsample_processed_path = f'./datasets/QA/{dataset}/{dataset}/test_subsample_processed.json'
    
    with open(test_subsample_path, 'r', encoding='utf-8') as test_subsample:
        with open(test_subsample_processed_path, 'w', encoding='utf-8') as json_file:
            for line in test_subsample:
                test_dic = json.loads(line)
                qid = test_dic['question_id']
                question = test_dic['question_text']
                answers = test_dic['answers_objects'][0]['spans']

                dic = {
                    'qid': qid,
                    'query_text': question,
                    'answers': answers
                }
                json_file.write(json.dumps(dic, ensure_ascii=False) + '\n')
    print('-' * 50)



# -----------------------------------------------------------------------------
# 4) get_multi_hop_dataset:
#    - Download & process multi-hop data (e.g., HotpotQA)
#    - Collect supporting passages as GT_chunk
#    - Sub-sample for final usage
# -----------------------------------------------------------------------------
def get_multi_hop_dataset(dataset):
    print('-' * 50)
    print(f'Processing dataset : {dataset}')

    if not os.path.exists(f'./datasets/QA/{dataset}/{dataset}/test_subsampled.jsonl'):
        subprocess.run(['bash', './download/processed_data.sh'], check=True)
        subprocess.run([
            'python',
            './processing_scripts/subsample_dataset_and_remap_paras.py',
            dataset,
            'test',
            str(1500)
        ], check=True)

    source_dir = f'./datasets/QA/{dataset}'
    dest_dir   = f'./datasets/QA/{dataset}/{dataset}'

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    if os.path.exists(source_dir):
        move_contents(source_dir, dest_dir)
        print(f"Moved contents: {source_dir} -> {dest_dir}")
    else:
        print(f"Source not found: {source_dir}")

    test_subsample_path = f'./datasets/QA/{dataset}/{dataset}/test_subsampled.jsonl'
    test_subsample_processed_path = f'./datasets/QA/{dataset}/{dataset}/test_subsample_processed.json'

    with open(test_subsample_path, 'r', encoding='utf-8') as test_subsample:
        with open(test_subsample_processed_path, 'w', encoding='utf-8') as json_file:
            for line in test_subsample:
                test_dic = json.loads(line)
                qid = test_dic['question_id']
                question = test_dic['question_text']
                answers = test_dic['answers_objects'][0]['spans']
                passage_dic = test_dic['contexts']

                gt_passages = []
                for passage in passage_dic:
                    if passage['is_supporting']:
                        gt_passages.append(passage['paragraph_text'])

                dic = {
                    'qid': qid,
                    'query_text': question,
                    'answers': answers,
                    'GT_chunk': gt_passages
                }
                json_file.write(json.dumps(dic, ensure_ascii=False) + '\n')

    folders_to_delete = [
        "./datasets/QA/2wikimultihopqa",
        "./datasets/QA/iirc",
        "./datasets/QA/musique"
    ]
    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted: {folder}")

    sub_sample_dataset(dataset)
    print('-' * 50)
    


# -----------------------------------------------------------------------------
# 5) get_other_dataset:
#    - An example for 'fiqa' or other long-form QA data
#    - Loads dataset, normalizes text, merges question->answer
#    - Saves a final JSON, then sub-samples
# -----------------------------------------------------------------------------
def get_other_dataset(dataset):
    qa_path = f'./datasets/QA/fiqa/fiqa/test_subsample_processed.json'
    ir_path = f'./datasets/IR/fiqa/fiqa/test_subsample_processed.json'
    os.makedirs(f'./datasets/QA/fiqa/fiqa', exist_ok=True)
    
    ds_test, ds_train = load_dataset('LLukas22/fiqa', split='test'), load_dataset('LLukas22/fiqa', split='train')
    df_test, df_train = map(pd.DataFrame, (ds_test, ds_train))
    
    df_test['question'], df_train['question'] = map(lambda df: df['question'].apply(normalize_text), (df_test, df_train))
    combined_dict = {**dict(zip(df_train['question'], df_train['answer'])), **dict(zip(df_test['question'], df_test['answer']))}
    
    test_queries, train_queries = map(lambda df: set(df['question'].apply(str.strip)), (df_test, df_train))
    query_id_list = set(pd.read_csv('./datasets/IR/fiqa/fiqa/qrels/test.tsv', sep='\t', header=None, names=['query_id', 'corpus_id', 'score'])['query_id'])
    
    with open('./datasets/IR/fiqa/fiqa/queries.jsonl', 'r', encoding='utf-8') as beir_queries, open(qa_path, 'w', encoding='utf-8') as output_file:
        test_cnt, train_cnt = 0, 0
        for line in beir_queries:
            data = json.loads(line)
            query, qid = normalize_text(data.get('text', '').strip()), data.get('_id')
            if qid in query_id_list and query in test_queries | train_queries:
                json.dump({'qid': qid, 'query_text': data['text'], 'answers': [combined_dict.get(query)]}, output_file, ensure_ascii=False)
                output_file.write("\n")
                test_cnt += query in test_queries
                train_cnt += query in train_queries
    
    print(f"test_cnt: {test_cnt}\ntrain_cnt: {train_cnt}\nSaved file: {qa_path}\n{'-'*50}")
    
    sub_sample_dataset(dataset)
    
    try:
        shutil.copy(qa_path, ir_path)
        print(f"File copied to: {ir_path}")
    except Exception as e:
        print(f"Error copying file: {e}")
    
    

# -----------------------------------------------------------------------------
# 6) sub_sample_dataset:
#    - Take a random sample of up to 500 lines from the processed file
#    - Overwrite the original file with just the sampled data
# -----------------------------------------------------------------------------
def sub_sample_dataset(dataset, seed=13370):
    if 'fiqa' in dataset:
        path = f'./datasets/QA/fiqa/fiqa/test_subsample_processed.json'
    else:
        path = f'./datasets/QA/{dataset}/{dataset}/test_subsample_processed.json'
    
    with open(path, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()

    total_lines = len(lines)
    local_rng = random.Random(seed)
    sampled_lines = local_rng.sample(lines, min(500, total_lines))
    sampled_data = [json.loads(line) for line in sampled_lines]
    
    with open(path, 'w', encoding='utf-8') as writer:
        for data in sampled_data:
            writer.write(json.dumps(data, ensure_ascii=False) + '\n')

    return sampled_data



if __name__ == '__main__':
    longform_datasets = ['LLukas22/fiqa']
    single_hop_datasets = ['nq', 'trivia', 'squad']
    multi_hop_datasets = ['hotpotqa']

    for dataset in multi_hop_datasets:
        get_multi_hop_dataset(dataset)
        
    for dataset in single_hop_datasets:
        if dataset == 'nq':
            get_nq_dataset(dataset)
        else:
            get_single_hop_dataset(dataset)
        
    for dataset in longform_datasets:
        get_other_dataset(dataset)

    zip_path = "./datasets/QA.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)