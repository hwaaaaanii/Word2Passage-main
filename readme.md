# README

## Overview
This project provides various Python scripts for IR (Information Retrieval) and QA (Question Answering) tasks.  
Use the scripts in the order described below to download datasets, generate pseudo data, rewrite queries, retrieve documents, and evaluate the results.

## Requirements
- **Python** 3.9
- Required libraries (install via `requirements.txt` or manually):
  pip install -r requirements.txt

--------------------------------------------------------------------------------

## File Descriptions

1. get_IR_dataset.py
   - Purpose: Downloads and prepares an IR dataset.
   - Usage:
     python get_IR_dataset.py
   - Example:
     python get_IR_dataset.py -dataset trec-covid
   - This command downloads and prepares the `trec-covid` dataset for IR.

--------------------------------------------------------------------------------

2. get_QA_dataset.py
   - Purpose: Downloads and prepares a QA dataset.
   - Usage:
     python get_QA_dataset.py
   - No additional arguments are required by default. Modify the script or its internal settings if needed.

--------------------------------------------------------------------------------

3. pseudo_generator.py
   - Purpose: Generates pseudo data for IR or QA tasks using a specified LLM.
   - Arguments (3):
     1) -dataset : e.g., trec-covid, nq, etc.
     2) -task : IR or QA
     3) -LLM : e.g., Llama3.1_8b, Llama3.1_70b, Qwen2.5_7b, Qwen2.5_72b
   - Usage:
     python pseudo_generator.py -dataset [DATASET_NAME] -task [IR/QA] -LLM [LLM_NAME]
   - Example:
     python pseudo_generator.py -dataset trec-covid -task IR -LLM Llama3.1_8b

--------------------------------------------------------------------------------

4. query_rewriter.py
   - Purpose: Rewrites queries to potentially improve retrieval performance.
   - Arguments (5):
     1) -dataset : e.g., trec-covid, nq, etc.
     2) -task : IR or QA
     3) -LLM : e.g., Llama3.1_8b
     4) -num : number of references
     5) -alpha : scaling factor (default: 30)
   - Usage:
     python query_rewriter.py \
       -dataset [DATASET_NAME] \
       -task [IR/QA] \
       -LLM [LLM_NAME] \
       -num [NUMBER_OF_REFERENCES] \
       -alpha [SCALING_FACTOR]
   - Example:
     python query_rewriter.py -dataset trec-covid -task IR -LLM Llama3.1_8b -num 5 -alpha 30

--------------------------------------------------------------------------------

5. eval_bm25_pytrec.py
   - Purpose: Evaluates IR results (BM25) using PyTREC.
   - Arguments (4):
     1) -dataset : e.g., trec-covid, etc.
     2) -task : IR
     3) -options : e.g., default, HyDE, MuGI, W2P (space-separated)
     4) -LLM : e.g., Llama3.1_8b
   - Usage:
     python eval_bm25_pytrec.py \
       -dataset [DATASET_NAME] \
       -task IR \
       -options [OPTIONS] \
       -LLM [LLM_NAME]
   - Example:
     python eval_bm25_pytrec.py -dataset trec-covid -task IR -options default HyDE MuGI W2P -LLM Llama3.1_8b

--------------------------------------------------------------------------------

6. QA_retriever.py
   - Purpose: Retrieves documents or passages for QA tasks with specified options.
   - Arguments (3):
     1) -dataset : e.g., nq
     2) -option : e.g., W2P, HyDE, MuGI, etc.
     3) -LLM : e.g., Llama3.1_8b
   - Usage:
     python QA_retriever.py \
       -dataset [DATASET_NAME] \
       -option [OPTION] \
       -LLM [LLM_NAME]
   - Example:
     python QA_retriever.py -dataset nq -option W2P -LLM Llama3.1_8b

--------------------------------------------------------------------------------

7. evaluation.py
   - Purpose: Evaluates QA results under different options (e.g., default, HyDE, MuGI, W2P).
   - Arguments (3):
     1) -dataset : e.g., nq
     2) -options : e.g., default HyDE MuGI W2P (space-separated)
     3) -LLM : e.g., Llama3.1_8b
   - Usage:
     python evaluation.py \
       -dataset [DATASET_NAME] \
       -options [OPTIONS] \
       -LLM [LLM_NAME]
   - Example:
     python evaluation.py -dataset nq -options default HyDE MuGI W2P -LLM Llama3.1_8b

--------------------------------------------------------------------------------

## Recommended Execution Flow

### IR Workflow Example
1) Get IR dataset
   python get_IR_dataset.py -dataset trec-covid
2) Generate pseudo data (optional)
   python pseudo_generator.py -dataset trec-covid -task IR -LLM Llama3.1_8b
3) Rewrite queries (optional)
   python query_rewriter.py -dataset trec-covid -task IR -LLM Llama3.1_8b -num 5 -alpha 30
4) Evaluate with BM25
   python eval_bm25_pytrec.py -dataset trec-covid -task IR -options default HyDE MuGI W2P -LLM Llama3.1_8b

### QA Workflow Example
1) Get QA dataset
   python get_QA_dataset.py
2) Generate pseudo data (optional)
   python pseudo_generator.py -dataset nq -task QA -LLM Llama3.1_8b
3) Retrieve documents for QA
   python QA_retriever.py -dataset nq -option W2P -LLM Llama3.1_8b
4) Evaluate QA results
   python evaluation.py -dataset nq -options default HyDE MuGI W2P -LLM Llama3.1_8b

--------------------------------------------------------------------------------

## Troubleshooting & Tips
- Library versions: If any compatibility issue arises, verify library versions via pip list and adjust your requirements.txt.
- File paths: Ensure correct absolute/relative paths for reading/writing data.
- Experimentation: Each script can accept different parameters (models, tasks, or options). Modify them as needed for your experiments.

--------------------------------------------------------------------------------

## License
Specify your project’s license here (e.g., MIT, Apache 2.0, etc.).
