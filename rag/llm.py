import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
import os

_llm_instance = None

def get_local_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading local LLM: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check for Mac (MPS) or CUDA or CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )

    # Increase model max length to avoid input truncation warnings.
    model.config.max_length = 4096
    tokenizer.model_max_length = 4096

    generation_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        clean_up_tokenization_spaces=False,
    )
    model.generation_config = generation_config

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        do_sample=generation_config.do_sample,
        pad_token_id=generation_config.pad_token_id,
        clean_up_tokenization_spaces=generation_config.clean_up_tokenization_spaces,
    )

    _llm_instance = HuggingFacePipeline(pipeline=pipe)
    return _llm_instance
