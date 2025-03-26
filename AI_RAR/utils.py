import yaml
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Dict, AnyStr, List

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# INTERNAL IMPORTS
from data_models import ResumeFeedback

# CACHE MANAGER CLASS
class CacheManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache = {}
            return cls._instance
    
    def set(self, key: str, value: Any) -> None:
        """SET A VARIABLE IN THE CACHE"""
        self._cache[key] = value
        print(f"---{key} ADDED IN CACHE---")
    
    def get(self, key: str, default = None) -> Any:
        """GET A VARIABLE FROM THE CACHE"""
        print(f"---GETTING {key} FROM CACHE---")
        return self._cache.get(key, default)

    def has(self, key: str) -> bool:
        """CHECK IF A VARIABLE EXISTS IN THE CACHE"""
        return key in self._cache
    
    def clear(self, key: str = None) -> None:
        """CLEAR A CATEGORY FROM THE CACHE"""
        if key:
            self._cache.pop(key, None)
            print(f"---{key} CLEARED FROM CACHE---")
        else:
            self._cache = {}
            print(f"---ALL CATEGORIES CLEARED FROM CACHE---")
    
    def append_to_list(self, key: str, value: Any) -> bool:
        """APPEND A VALUE TO A LIST IN THE CACHE"""
        if key in self._cache:
            if isinstance(self._cache[key], list):
                self._cache[key].append(value)
                print(f"---APPENDED VALUE TO {key} IN CACHE---")
                return True
            else:
                print(f"---ERROR: {key} IS NOT A LIST---")
                return False
        else:
            self._cache[key] = [value]
            print(f"---CREATED NEW LIST WITH VALUE IN {key}---")
            return True

    def remove_from_list(self, key: str, value: Any) -> bool:
        """REMOVE A VALUE FROM A LIST IN THE CACHE"""
        if key in self._cache and isinstance(self._cache[key], list):
            try:
                self._cache[key].remove(value)
                print(f"---REMOVED VALUE FROM {key} IN CACHE---")
                return True
            except ValueError:
                print(f"---VALUE NOT FOUND IN {key}---")
                return False
        else:
            print(f"---ERROR: {key} NOT FOUND OR NOT A LIST---")
            return False

# FUNCTION TO LOAD PROMPTS
def load_prompts(path: Path) -> dict:
    print(f"---LOADING PROMPTS FROM {path}---")
    with open(path, 'r') as file:
        prompts = yaml.safe_load(file)
    return prompts

# FUNCTION TO LOAD AND PROCESS JOB DESCRIPTION TXT FILES FROM DIRECTORY
def process_directory(directory_path, file_content):
    results = []
    
    if file_content == "job_description":
        # Process TXT files for job descriptions
        for file in os.listdir(directory_path):
            if file.endswith(".txt"):
                file_path = os.path.join(directory_path, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    results.append({
                        "name": file,
                        "content": content
                    })
    elif file_content == "resume":
        # Process PDF files for resumes
        for file in os.listdir(directory_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(directory_path, file)
                # Use your PDF processing logic here
                # For example:
                try:
                    # Extract text from PDF
                    loader = PyPDFLoader(str(file_path))
                    pdf_documents = loader.load()

                    # Combine all pages into a single document
                    full_text = "\n".join([doc.page_content for doc in pdf_documents])

                    results.append(Document(
                        page_content=full_text,
                        metadata={"source": file}
                    ))
                except Exception as e:
                    raise RuntimeError(f"Error reading resume file {file_path.name}: {str(e)}")
    
    return results
# def process_directory(directory_path: str, file_content: Literal["job_description", "resume"]):
    
#     # CHECK IF FILE CONTENT IS JOB DESCRIPTION OR RESUME
#     if file_content == "job_description":
#         job_descriptions = []

#         # BEGIN PROCESSING EACH .txt files
#         job_files = list(Path(directory_path).glob("**/*.txt"))

#         if job_files:
#             for job_file in job_files:
#                 try:
#                     with open(job_file, "r", encoding='utf-8') as f:
#                         job_description_content = f.read()

#                     job_descriptions.append({
#                             "name": job_file.name,
#                             "content": job_description_content

#                         })
#                 except Exception as e:
#                     raise RuntimeError(f"Error reading job description file {job_file.name}: {str(e)}")

#             return job_descriptions
                
#         else:
#             raise RuntimeError("No .txt files found in the directory")
    
#     elif file_content == "resume":
#         document_resumes = []

#         for file_path in Path(directory_path).glob("**/*.pdf"):
#             try:
#                 # Extract text from PDF
#                 loader = PyPDFLoader(str(file_path))
#                 pdf_documents = loader.load()

#                 # Combine all pages into a single document
#                 full_text = "\n".join([doc.page_content for doc in pdf_documents])

#                 document_resumes.append(Document(
#                     page_content=full_text,
#                     metadata={"source": file_path.name}
#                 ))
#             except Exception as e:
#                 raise RuntimeError(f"Error reading resume file {file_path.name}: {str(e)}")
            
#         return document_resumes
    
#     else:
#         raise RuntimeError("Invalid file content type.")

# FUNCTION TO FLATTEN RESUME FEEDBACK RANKINGS AND JOB DESCRIPTIONS
def flatten(all_rankings: Dict[AnyStr, List[ResumeFeedback]], jobs: List[Dict[AnyStr, AnyStr]]):
    try:
        flattened_string = ""
        num_jobs = len(jobs)

        for job in jobs:
            num_jobs -= 1
            print(f'---FLATTENING {job.get("name")} ({num_jobs} LEFT)---')
            flattened_job_description = f"# Job Openings: \nFileName: {job.get('name')}\n{job.get('content')}\n\n# Resume Ranking and Analysis:"
            flattened_candidates = ""
            for job_name, ranking in all_rankings.items():
                if job_name == job.get("name"):
                    for idx, candidate in enumerate(ranking):
                        flattened_candidates += f"""\n## Rank {idx+1}\n## Candidate Name: {candidate.candidate_name}
## Analysis: {candidate.analysis}
## Scores: {candidate.scores}
## Total Score: {candidate.total_score}
## Key Strengths: {candidate.key_strengths}
## Areas for Improvement: {candidate.areas_for_improvement}\n"""
                        
                    flattened_string += f"{flattened_job_description}\n{flattened_candidates}\n"

        return flattened_string
    except Exception as e:
        raise RuntimeError(f"Error flattening rankings and job descriptions: {str(e)}")

# FUNCTION TO SETUP VECTOR STORE
def setup_vector_store(cache_manager: CacheManager):
    print(f"---SETTING UP VECTOR STORE---")

    # SET EMBEDDING MODEL TO CACHE IF NOT ALREADY SET
    if not cache_manager.has("embedding_model"):
        embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HF_API_KEY"),
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        cache_manager.set("embedding_model", embedding_model)
    
    embedding_model = cache_manager.get("embedding_model")

    # SET CHROMA VECTOR STORE TO CACHE IF NOT ALREADY SET
    if not cache_manager.has("vector_store"):
        vector_store = Chroma(
            collection_name="resume_ranking",
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )
        cache_manager.set("vector_store", vector_store)

        return vector_store

# FUNCTION TO PROCESS INDIVIDUAL TXT FILES
def process_txt(txt_file):
    try:
        job_description_content = txt_file.getvalue().decode("utf-8")
        job_descriptions = [{
            "name": txt_file.name,
            "content": job_description_content
        }]
        print(f"Job description uploaded: {txt_file.name}")

        return job_descriptions
    except Exception as e:
        raise RuntimeError(f"Error reading job description file: {str(e)}")

# FUNCTION TO PROCESS PDFS
def process_pdfs(pdf_files):
    """Process multiple PDF files and return a list of Documents"""
    documents = []
    
    for pdf_file in pdf_files:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, pdf_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Extract text from PDF
        loader = PyPDFLoader(temp_file_path)
        pdf_documents = loader.load()
        
        # Combine all pages into a single document
        full_text = "\n".join([doc.page_content for doc in pdf_documents])
        
        documents.append(Document(
            page_content=full_text,
            metadata={"source": pdf_file.name}
        ))
        
        temp_dir.cleanup()
    
    return documents