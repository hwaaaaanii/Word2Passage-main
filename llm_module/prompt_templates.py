def W2P_generate_prompt(query: str) -> str:
    """
    Generates a passage, a sentence, and a list of words relevant to the query.

    Parameters:
    - query (str): The input question.

    Returns:
    - str: A formatted prompt enforcing a structured JSON output.
    """
    PROMPT = f"""
Generate a passage, a sentence, and words that answer the given QUERY. 
Terms that are important for answering the QUERY should frequently appear in the generation of the passage, the sentence, and words.

### Definition:
**passage**: Answer the given QUERY in a passage perspective by generating an informative and clear passage.
**sentence**: Answer the given QUERY in a sentence perspective by generating a knowledge-intensive sentence.
**word**: Answer the given QUERY in a word perspective by generating a list of words.

### QUERY:
{query}

### FINAL OUTPUT JSON FORMAT (strictly follow this structure):
{{
"passage": "Your passage here",
"sentence": "Your sentence here",
"word": [Your words here],
}}
(From here on, only produce the final output in the specified JSON format.)
"""
    return PROMPT



def MuGI_passage_prompt(query: str) -> str:
    """
    Generates a relevant passage for the given query using the MuGI approach.

    Parameters:
    - query (str): The input question.

    Returns:
    - str: A formatted MuGI passage generation prompt.
    """
    PROMPT = f'''
Generate one passage that is relevant to the following query: '{query}'. 
The passage should be concise,informative, and clear'''
    return PROMPT



def HyDE_passage_prompt(query: str) -> str:
    """
    Generates a hypothetical document (HyDE) passage based on a given query.

    Parameters:
    - query (str): The input question.

    Returns:
    - str: A formatted HyDE passage generation prompt.
    """
    PROMPT = f"""
    Please write a passage to answer the question
    Question: [{query}]
    Passage: """
    return PROMPT



def llm_eval_prompt(query: str, gt: str, pd: str) -> str:
    """
    Generates a prompt for evaluating an AI-generated answer.

    Parameters:
    - query (str): The input question.
    - gt (str): The ground truth (golden answer).
    - pd (str): The AI-generated answer.

    Returns:
    - str: A formatted evaluation prompt.
    """
    PROMPT = f"""
You are an evaluation tool. Just answer with {{Yes}} or {{No}}. 
Here is a question, a golden answer, and an AI-generated answer. 
Judge whether the AI-generated answer is correct according to the question and the golden answer.
Answer strictly with {{Yes}} or {{No}}.

Question: {query}
Golden Answer: {gt}
Generated Answer: {pd}
Response: 
"""
    return PROMPT



def answer_generation_prompt(query: str, retrieved_documents: str) -> str:
    """
    Generates a prompt for answering a query using retrieved documents.

    Parameters:
    - query (str): The input question.
    - retrieved_documents (str): Contextual documents retrieved for answering the query.

    Returns:
    - str: A formatted generation prompt.
    """
    PROMPT = f"""
Answer the given QUERY based on the provided CONTEXT and your knowledge.

QUERY: {query}

CONTEXT: {retrieved_documents}

Only generate the answer in JSON format. Provide no explanations or additional information.
The answer should be concise.

Strict JSON format:
{{"Answer": "your answer"}}
"""
    return PROMPT



def query_type_classification_prompt(query):
    PROMPT = f"""
    You are given a dataset containing queries categorized into different types. Here are some examples:

    Query Type: description
    - Query: causes of inflamed pelvis
    - Query: name the two types of cells in the cortical collecting ducts and describe their function

    Query Type: numeric
    - Query: military family life consultant salary
    - Query: average amount of money spent on entertainment per month

    Query Type: location
    - Query: what is the biggest continent
    - Query: where is trinidad located

    Query Type: entity
    - Query: what kind of plants grow in oregon?
    - Query: what are therapy animals

    Query Type: person
    - Query: who is guardian angel cassiel
    - Query: interstellar film cast

    Now, classify the following query into one of the above categories.
    Choose only one of the following categories:
    [description, numeric, location, entity, person]

    Query: {query}

    ### OUTPUT FORMAT
    Query Type: your answer (must be one of the categories listed above)
    """

    return PROMPT
