from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

from pathlib import Path
import time
import sys
import os
import yaml
from dotenv import load_dotenv

# INTERNAL IMPORTS
from data_models import ResumeFeedback, CrossJobMatchResult
from utils import CacheManager, load_prompts

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# INITIALIZE CACHE MANAGER
cache_manager = CacheManager()

# PATH TO PROMPTS
PROMPTS_PATH = Path("prompts.yaml")

# CACHE ALL IMPORTS AS A DICTIONARY
cache_imports = {}

# LIST ALL MODELS AVAILABLE FROM GROQ
GROQ_MODELS = [
    "llama-3.3-70b-versatile", 
    "llama-3.1-8b-instant",
    "qwen-qwq-32b",
    "qwen-2.5-32b"
]

# LIST ALL MODELS AVAILABLE FROM MISTRALAI
MISTRALAI_MODELS = [
    "mistral-large-latest",
    "ministral-8b-latest"
]

# FUNCTION TO INITIALIZE LLM MODEL
def initialize_llm(model_name: str) -> BaseChatModel:

    # CHECK IF THE MODEL EXIST IN CACHE
    cached_model = cache_manager.get(model_name)
    if cached_model:
        return cached_model

    # CHECK THE TYPE OF THE LLM MODEL TO USE
    if model_name in GROQ_MODELS:
        print(f"---LOADING {model_name} FROM GROQ---")
        groq_model = ChatGroq(
                model_name=model_name,
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY")
            )

        # CACHE THE GROQ MODEL
        cache_manager.set(model_name, groq_model)
        
        return groq_model
    
    elif model_name in MISTRALAI_MODELS:
        print(f"---LOADING {model_name} FROM MISTRALAI---")
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Your existing agent creation code
                mistral_model = ChatMistralAI(
                    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                    model="mistral-large-latest"
                )
                # CACHE THE MISTRAL MODEL
                cache_manager.set(model_name, mistral_model)
                
                return mistral_model
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise e


# CREATE THE RESUME ANALYZER RERANKER AGENT
def create_rar_agent() -> RunnableSerializable:
    try:
        # SET PROMPTS IF NOT IN CACHE
        if not cache_manager.has("agent_prompts"):
            cache_manager.set("agent_prompts", load_prompts(PROMPTS_PATH))

        rar_agent_prompt = cache_manager.get("agent_prompts")["rar_agent_prompt"]

        # INITIALIZE THE LLM MODEL
        rar_llm = initialize_llm("llama-3.3-70b-versatile")

        # ATTACH DATA MODEL TO AGENT
        rar_llm_with_structured_output = rar_llm.with_structured_output(ResumeFeedback)

        # PREPARE CHAT PROMPT TEMPLATE
        rar_agent_sys_prompt = ChatPromptTemplate.from_template(rar_agent_prompt)

        # CHAIN THE PROMPT TEMPLATE WITH THE RAR AGENT
        rar_agent_chain = rar_agent_sys_prompt | rar_llm_with_structured_output

        print(f"---RESUME ANALYZER-RERANKER AGENT CREATED---")

        return rar_agent_chain
    except Exception as e:
        print(f"---ERROR IN CREATING RAR AGENT: {e}---")
        raise RuntimeError(f"FAILED TO CREATE RAR AGENT: {e}")

# CREATE THE CROSS JOB COMPARISON AGENT
def create_cjc_agent() -> RunnableSerializable:
    try:
        # SET PROMPTS IF NOT IN CACHE
        if not cache_manager.has("agent_prompts"):
            cache_manager.set("agent_prompts", load_prompts(PROMPTS_PATH))

        cjc_agent_prompt = cache_manager.get("agent_prompts")["cjc_agent_prompt"]

        # INITIALIZE THE LLM MODEL
        cjc_llm = initialize_llm('mistral-large-latest')

        # ATTACH DATA MODEL TO AGENT
        cjc_llm_with_structured_output = cjc_llm.with_structured_output(CrossJobMatchResult)

        # PREPARE CHAT PROMPT TEMPLATE
        cjc_agent_sys_prompt = ChatPromptTemplate.from_template(cjc_agent_prompt)

        # CHAIN THE PROMPT TEMPLATE WITH THE RAR AGENT
        cjc_agent_chain = cjc_agent_sys_prompt | cjc_llm_with_structured_output

        print(f"---CROSS JOB COMPARISON AGENT CREATED---")

        return cjc_agent_chain
    except Exception as e:
        print(f"---ERROR IN CREATING CJC AGENT: {e}---")
        raise RuntimeError(f"FAILED TO CREATE CJC AGENT: {e}")