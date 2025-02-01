from sentence_transformers import SentenceTransformer
import numpy as np
import pyarrow as pa
import yaml
import os
from pathlib import Path

# FUNCTION TO LOAD THE EMBEDDING MODEL
def load_embedding_model(hf_token: str, model_name: str = 'nomic-ai/nomic-embed-text-v1.5') -> SentenceTransformer:
    model = SentenceTransformer(model_name, trust_remote_code=True, token=hf_token)
    return model

# FUNCTION TO STRUCTURE THE CONTEXT FROM THE RETRIEVED FILES
def structure_context(data, index: np.ndarray) -> str:
    # CONVERT PYARROW TABLE TO DICTIONARY FOR EASIER ACCESS
    data = data.to_pylist()
    
    # DICTIONARY TO GROUP ENTRIES BY FILENAME {FILENAME: [(PAGES, CONTENT), ...]}
    file_groups = {}
    # LIST TO PRESERVE FILENAME ORDER
    file_order = []
    
    for i in index:
        filename = data[i].get("fileName")
        pages = data[i].get("pageNumbers")
        content = data[i].get("suggestedText").replace('  ', ' ').strip()

        if filename not in file_groups:
            file_groups[filename] = []
            file_order.append(filename)
            
        file_groups[filename].append((pages, content))

    # BUILD CONTEXT WITH PROPER GROUPING
    context = ""
    for filename in file_order:
        context += f"### Source Title: {filename}\n"
        
        for pages, content in file_groups[filename]:
            context += f"#### From Page Number(s): {pages}\n"
            context += f"**Relevant Content:**\n{content}\n\n"

    return context

# FUNCTION TO LOAD SYS PROMPTS
def load_sys_prompts(prompts_path: Path) -> dict:
    with open(prompts_path, "r") as file:
        sys_prompt = yaml.safe_load(file)
    
    return sys_prompt



