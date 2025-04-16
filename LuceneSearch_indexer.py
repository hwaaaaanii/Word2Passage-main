'''
This code requires about 20GB
Make sure retain the necessary disk space before running.
The script converts a TSV corpus to JSONL format and indexes it using Pyserini.
'''

import argparse
import json
import pandas as pd
import os
import subprocess

parser = argparse.ArgumentParser(description="Generate pseudo passages")
parser.add_argument('-corpus_name', type=str, required=True)
args = parser.parse_args()

if args.corpus_name == 'psgs_w100':
    tsv_path = f'./datasets/data/{args.corpus_name}.tsv'  
    output_jsonl_path = f'./datasets/data/{args.corpus_name}.jsonl'  
    index_output_path = f'./datasets/data/lucene_index/{args.corpus_name}' 
else:
    tsv_path = f'./datasets/user_data/corpus/{args.corpus_name}.tsv'  
    output_jsonl_path = f'./datasets/user_data/corpus/{args.corpus_name}.jsonl'  
    index_output_path = f'./datasets/user_data/lucene_index/{args.corpus_name}' 


if not os.path.exists(output_jsonl_path):
    print(f"Loading corpus...")
    df = pd.read_csv(tsv_path, sep='\t')  
    print(f'Writing corpus into josnl format...')
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json_obj = {
                'id': str(row['id']),  
                'contents': row['text'] 
            }
            print(json_obj)
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"✅ JSONL file saved at {output_jsonl_path}")
else:
    print(f"✅ JSONL file already exists at {output_jsonl_path}")

cmd = [
    'python', '-m', 'pyserini.index.lucene',
    '--collection', 'JsonCollection',
    '--input', os.path.dirname(output_jsonl_path),
    '--index', index_output_path,
    '--generator', 'DefaultLuceneDocumentGenerator',
    '--threads', '4',  
    '--storePositions', '--storeDocvectors', '--storeRaw'
]

subprocess.run(cmd, check=True)

print(f"✅ Lucene index created at {index_output_path}")