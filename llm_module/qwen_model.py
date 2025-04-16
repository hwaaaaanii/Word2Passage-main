import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random

# Set random seeds for reproducibility
def set_seed():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

set_seed()

def load_qwen_model(llm_arg: str):
    """
    Dynamically loads the Qwen model (7B or 72B) based on the provided llm_arg.

    Parameters
    ----------
    llm_arg : str
        Possible values:
        - 'Qwen2.5_7b'
        - 'Qwen2.5_72b'
        (or any naming scheme you choose)
    """
    global model
    global tokenizer

    if llm_arg == "Qwen2.5_7b":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
    elif llm_arg == "Qwen2.5_72b":
        model_name = "Qwen/Qwen2.5-72B-Instruct"
    else:
        raise ValueError(f"Unsupported LLM option: {llm_arg}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    print(f"[qwen_model.py] Model loaded for: {llm_arg} -> {model_name}")

# System prompts for different tasks
W2P_SYS_PROMPT = "You are tasked with generating words, a sentence, a passage to be used for query augmentation."
MuGI_SYS_PROMPT = "You are PassageGenGPT, an AI capable of generating concise, informative, and clear pseudo passages on specific topics."
ANSWER_SYS_PROMPT = """You are an assistant for answering questions.
Provide a brief answer which should be a short form.
If you don't know the answer, just say "null" Don't make up an answer."""



def generate_response(messages, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=1.0):
    """
    Generates a response using the given messages as input.

    Parameters:
    - messages (list of dict): List of messages in [{"role": ..., "content": ...}, ...] format.
    - max_new_tokens (int, optional): Maximum number of new tokens to generate. Default is 512.
    - do_sample (bool, optional): Whether to use sampling during generation. Default is True.
    - temperature (float, optional): Controls randomness. Higher values lead to more random responses. Default is 1.0.
    - top_p (float, optional): Controls nucleus sampling. Lower values limit diversity. Default is 1.0.

    Returns:
    - str: The generated response.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )
    prompt_length = model_inputs.input_ids.shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response



def generate_W2P(formatted_prompt):
    messages = [
        {"role": "system", "content": W2P_SYS_PROMPT},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(messages, max_new_tokens=1000, do_sample=True, temperature=1.4, top_p=0.9)



def generate_HyDE(formatted_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(messages, max_new_tokens=1000, do_sample=True)



def generate_MuGI(formatted_prompt):
    messages = [
        {"role": "system", "content": MuGI_SYS_PROMPT},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(messages, max_new_tokens=1000, do_sample=True)



def generate_answer(formatted_prompt):
    messages = [
        {"role": "system", "content": ANSWER_SYS_PROMPT},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(messages, max_new_tokens=500, do_sample=False)



def generate_answer_do_sample(formatted_prompt):
    messages = [
        {"role": "system", "content": ANSWER_SYS_PROMPT},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(messages, max_new_tokens=500, do_sample=True)



def generate_query_label(formatted_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(messages, max_new_tokens=1000, do_sample=True)
