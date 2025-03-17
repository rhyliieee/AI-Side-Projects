# app.py
import os
import tempfile
import streamlit as st
from pathlib import Path
import pandas as pd
import time
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.storage import LocalFileStore
import langgraph
from langgraph.graph import StateGraph
# from langgraph.prebuilt import ToolExecutor, tools
from langgraph.graph import END, StateGraph
from typing import TypedDict, List, Dict, Any, Annotated
import json
from dotenv import load_dotenv
import asyncio
from playwright.async_api import async_playwright
import requests

from data_models import ResumeFeedback
from utils import load_prompts

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Resume Ranking System",
    page_icon="📄",
    layout="wide"
)

# Application title and description
st.title("🤖 AI-Powered Resume Ranking System")
st.markdown("""
This application analyzes resumes/CVs and ranks them based on a job posting using AI.
""")

# MODEL SELECTION
groq_model_list = [
    "llama-3.3-70b-versatile", 
    "qwen-qwq-32b",
    "mixtral-8x7b-32768",
]

# INDIVIDUAL MODEL
model_list = ["mistral-large-latest", "gemini-2.0-flash"] + groq_model_list

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    # groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    st.markdown("---")
    st.header("Models")
    llm_model = st.selectbox(
        "Select LLM Model", 
        model_list
    )
    st.markdown("---")
    st.header("About")
    st.info("This app uses Groq API, LangChain, LangGraph, and ChromaDB to analyze and rank resumes.")

# Define LLM
@st.cache_resource
def get_llm(model_name):

    # CHECK MODEL NAME TO ASSIGN TO CORRECT INFERENCE PROVIDER  
    if model_name in groq_model_list:
        return ChatGroq(
            model_name=model_name,
            temperature=0.2,
            api_key=os.getenv("GROQ_API_KEY")
        )
    elif model_name == "mistral-large-latest":
        return ChatMistralAI(
            model_name=model_name,
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.2
        )
    else:
        return ChatGoogleGenerativeAI(
            model_name=model_name,
            api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
            temperature=0.2
        )

llm = get_llm(llm_model)

# Resume processing functions
def process_pdfs(pdf_files):
    """Process multiple PDF files and return a list of Documents"""
    documents = []
    
    with st.spinner("Processing PDF files..."):
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

def process_directory(directory_path):
    """Process all PDF files in a directory and return a list of Documents"""
    documents = []
    
    with st.spinner(f"Processing PDF files from directory..."):
        for file_path in Path(directory_path).glob("**/*.pdf"):
            # Extract text from PDF
            loader = PyPDFLoader(str(file_path))
            pdf_documents = loader.load()
            
            # Combine all pages into a single document
            full_text = "\n".join([doc.page_content for doc in pdf_documents])
            
            documents.append(Document(
                page_content=full_text,
                metadata={"source": file_path.name}
            ))
    
    return documents

# Vector store setup
@st.cache_resource
def get_vectorstore():
    # Initialize embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    
    # Setup caching for embeddings
    # store = LocalFileStore("./cache/embeddings")
    # cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    #     underlying_embeddings=embeddings,
    #     document_embedding_cache=store,
    #     namespace=embeddings.model
    # )
    
    # Create a temporary vector store
    vectorstore = Chroma(
        collection_name="resume_ranking",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# LangGraph agent definition
class AgentState(TypedDict):
    job_description: str
    resumes: List[Dict[str, Any]]
    current_resume_index: int
    scores: List[Dict[str, Any]]
    analysis: str
    final_ranking: List[Dict[str, Any]]

def initialize_state(job_description: str, resumes: List[Dict[str, Any]]) -> AgentState:
    return {
        "job_description": job_description,
        "resumes": resumes,
        "current_resume_index": 0,
        "scores": [],
        "analysis": "",
        "final_ranking": []
    }

def analyze_resume(state: AgentState) -> AgentState:
    """Analyze the current resume and score it based on job description"""
    print(f"---STARTING TO ANALYZE RESUME---")
    if state["current_resume_index"] >= len(state["resumes"]):
        return state
    
    current_resume = state["resumes"][state["current_resume_index"]]
    job_description = state["job_description"]

    # INITIALIZE PROMPTS
    prompts = load_prompts(Path("prompts.yaml"))

    # ASSIGN DATA MODEL FOR THE LLM
    structured_agent_analyzer = llm.with_structured_output(ResumeFeedback)
    
    prompt = ChatPromptTemplate.from_template(prompts["resume_agent_prompt"])
    
    chain = prompt | structured_agent_analyzer

    print("---INVOKING CHAIN---")
    
    try:
        result = chain.invoke({
            "job_description": job_description,
            "resume_content": current_resume["content"]
        })

        print(f"---AGENT RESULT: {current_resume['name']}\n{result}\n---")
        
        # Convert result to a dictionary
        result_dict = result.model_dump()
        
        # Add resume name to the result
        result_dict["resume_name"] = current_resume["name"]

        # CONVERT RESULT BACK TO ResumeFeedback OBJECT
        # result = ResumeFeedback(**result_dict)
        
        # Update the state
        state["scores"].append(result_dict)
        
    except Exception as e:
        st.error(f"Error analyzing resume {current_resume['name']}: {str(e)}")
        # Add a minimal score object in case of failure
        state["scores"].append({
            "resume_name": current_resume["name"],
            "analysis": f"Error analyzing resume: {str(e)}",
            "scores": {
                "skills_match": 0,
                "experience_relevance": 0,
                "education_fit": 0,
                "overall_impression": 0
            },
            "total_score": 0,
            "key_strengths": [],
            "areas_for_improvement": ["Could not analyze resume"]
        })
        print(f"---ERROR ANALYZING RESUME {current_resume['name']}: {str(e)}---")
    
    state["current_resume_index"] += 1
    return state

def create_final_ranking(state: AgentState) -> AgentState:
    """Create the final ranking of resumes based on scores"""
    print(f"---PERFORMING RANKING OF RESUMEs---")

    if not state["scores"]:
        return state
    
    # Sort the scores by total_score in descending order
    ranked_resumes = sorted(
        state["scores"], 
        key=lambda x: x["total_score"],	 
        reverse=True
    )
    
    # Add rank to each resume
    for i, resume in enumerate(ranked_resumes):
        resume["rank"] = i + 1
    
    state["final_ranking"] = ranked_resumes
    print(f"---RESUMES RANKED---")
    return state

def should_continue_analyzing(state: AgentState) -> str:
    """Determine if we should continue analyzing resumes or create the final ranking"""
    if state["current_resume_index"] < len(state["resumes"]):
        return "analyze_resume"
    else:
        return "create_final_ranking"

# Create the workflow graph
def create_resume_ranking_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_resume", analyze_resume)
    workflow.add_node("create_final_ranking", create_final_ranking)
    
    # Add edges
    workflow.add_conditional_edges(
        "analyze_resume",
        should_continue_analyzing,
        {
            "analyze_resume": "analyze_resume",
            "create_final_ranking": "create_final_ranking"
        }
    )
    
    workflow.add_edge("create_final_ranking", END)
    
    # Set the entry point
    workflow.set_entry_point("analyze_resume")
    
    return workflow.compile()

# Main application layout
tab1, tab2 = st.tabs(["Resume Ranking", "Results Explanation"])

with tab1:
    # Job description input
    st.header("Step 1: Enter Job Description")
    # job_description = st.text_area(
    #     "Paste the job description here:",
    #     height=200,
    #     help="Enter the full job description to compare resumes against."
    # )
    
    # @st.cache_resource   #Cache for 1 hour
    async def scrape_job_listings(url):
        """Scrape job listings from the provided URL using Playwright"""
        job_links = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(url, wait_until="domcontentloaded")
                
                # Wait for job listings to load
                await page.wait_for_selector('.awsm-job-listing-item.awsm-grid-item', timeout=10000)
                
                # Get all job listing items
                job_items = await page.query_selector_all('.awsm-job-listing-item.awsm-grid-item')
                
                for item in job_items:
                    # Find the link in each job item
                    link_element = await item.query_selector('a')
                    if link_element:
                        job_url = await link_element.get_attribute('href')
                        job_title = await link_element.text_content()
                        job_links.append({
                            'title': job_title.strip(),
                            'url': job_url
                        })
                        print(f"\n\n---JOB LINK: {job_links}---\n\n")
            except Exception as e:
                st.error(f"Error scraping job listings: {str(e)}")
            finally:
                await browser.close()
                
        return job_links
    
    async def scrape_job_description(url):
        "USE JINA AI TO CONVERT A GIVEN LINK INTO A MARKDOWN TEXT"
        job_description = ""
        try:
            # url = "https://r.jina.ai/https://example.com"
            headers = {
                "Authorization": "Bearer jina_28012d41c50549dcaaf9966f76018193D9LBJZPO-m68G9a693l3sNYIbIy9",
                "X-No-Cache": "true",
                "X-Retain-Images": "none",
                "X-Target-Selector": ".awsm-job-content"
            }

            response = await requests.get(url, headers=headers)
            job_description = response.text
            print(f"---JOB DESCRIPTION: {job_description}---")
            return job_description
        except Exception as e:
            st.error(f"Error scraping job description: {str(e)}")
            print(f"---ERROR SCRAPING JOB DESCRIPTION: {str(e)}---")

    # Default URL
    default_url = "https://direcbusiness.com/careers/"
    job_site_url = st.text_input("Job listing URL", value=default_url)
    
    if st.button("Fetch Job Listings"):
        with st.spinner("Fetching job listings..."):
            job_listings = asyncio.run(scrape_job_listings(job_site_url))
            if job_listings:
                st.session_state.job_listings = job_listings
                st.success(f"Found {len(job_listings)} job listings!")
            else:
                st.error("No job listings found. Please check the URL or website structure.")
    
    # Display job listings and let user select
    job_description = ""
    if "job_listings" in st.session_state and st.session_state.job_listings:
        selected_job = st.selectbox(
            "Select a job posting to analyze resumes against:",
            options=st.session_state.job_listings,
            format_func=lambda x: x['title']
        )
        
        if selected_job and st.button("Fetch Job Description"):
            with st.spinner("Fetching job description..."):
                job_description_md = asyncio.run(scrape_job_description(selected_job['url']))
                
                # Store in session state
                st.session_state.job_description = job_description_md
                
                # Display job description
                with st.expander("Job Description", expanded=True):
                    st.write(job_description_md)
    
    # Use the stored job description if available
    if "job_description" in st.session_state:
        job_description = st.session_state.job_description

    # File upload options
    st.header("Step 2: Upload Resumes")
    upload_option = st.radio(
        "Choose upload method:",
        ("Upload individual PDF files", "Upload a folder of PDF files")
    )
    
    documents = []
    
    if upload_option == "Upload individual PDF files":
        uploaded_files = st.file_uploader(
            "Upload resume PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} files uploaded")
            documents = process_pdfs(uploaded_files)
            
    else:  # Upload folder option
        upload_dir = st.text_input(
            "Enter the path to directory containing resume PDFs:",
            help="Example: /path/to/resumes"
        )
        
        if upload_dir and os.path.isdir(upload_dir):
            st.success(f"Directory found: {upload_dir}")
            documents = process_directory(upload_dir)
            st.info(f"{len(documents)} PDF files found in the directory")
        elif upload_dir:
            st.error(f"Directory not found: {upload_dir}")
    
    # Process button
    if st.button("Rank Resumes", disabled=not (job_description and documents)):
        if not job_description:
            st.error("Please enter a job description")
        elif not documents:
            st.error("Please upload at least one resume")
        else:
            with st.spinner("Analyzing and ranking resumes... This may take a few minutes."):
                # Prepare data for the agent
                resumes = []
                for doc in documents:
                    resumes.append({
                        "name": doc.metadata["source"],
                        "content": doc.page_content
                    })
                
                # Initialize the agent
                resume_agent = create_resume_ranking_agent()
                
                # Execute the agent
                initial_state = initialize_state(job_description, resumes)
                final_state = resume_agent.invoke(initial_state)
                
                # Store results in session state for the results tab
                st.session_state.ranking_results = final_state["final_ranking"]
                st.session_state.job_description = job_description
                
                # Display summary
                st.success("Analysis complete! View detailed results in the Results Explanation tab.")
                
                # Create a DataFrame for the results
                results_df = pd.DataFrame([
                    {
                        "Rank": result["rank"],
                        "Resume": result["resume_name"],
                        "Total Score": result["total_score"],
                        "Skills Match": result["scores"]["skills_match"],
                        "Experience": result["scores"]["experience_relevance"],
                        "Education": result["scores"]["education_fit"],
                        "Impression": result["scores"]["overall_impression"]
                    }
                    for result in final_state["final_ranking"]
                ])
                
                # Display the results table
                st.header("Ranking Results")
                st.dataframe(
                    results_df.style.highlight_max(subset=["Total Score"], color="lightgreen"),
                    use_container_width=True
                )

with tab2:
    if "ranking_results" in st.session_state:
        st.header("Detailed Ranking Results")
        
        # Display job description
        with st.expander("Job Description", expanded=True):
            st.write(st.session_state.job_description)
        
        # Create tabs for each resume
        ranked_resumes = st.session_state.ranking_results
        resume_tabs = st.tabs([f"#{r['rank']} - {r['resume_name']} (Score: {r['total_score']})" for r in ranked_resumes])
        
        for i, tab in enumerate(resume_tabs):
            resume_data = ranked_resumes[i]
            with tab:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Analysis")
                    st.write(resume_data["analysis"])
                
                with col2:
                    # Score breakdown
                    st.subheader("Score Breakdown")
                    scores = resume_data["scores"]
                    
                    # Create a DataFrame for the scores
                    score_data = {
                        "Category": ["Skills Match", "Experience Relevance", "Education Fit", "Overall Impression", "Total"],
                        "Score": [
                            scores["skills_match"],
                            scores["experience_relevance"],
                            scores["education_fit"],
                            scores["overall_impression"],
                            resume_data["total_score"]
                        ],
                        "Max": [30, 40, 20, 10, 100]
                    }
                    score_df = pd.DataFrame(score_data)
                    
                    # Display as a table
                    st.dataframe(score_df, use_container_width=True, hide_index=True)
                    
                    # Display key strengths
                    st.subheader("Key Strengths")
                    for strength in resume_data["key_strengths"]:
                        st.markdown(f"✅ {strength}")
                    
                    # Display areas for improvement
                    st.subheader("Areas for Improvement")
                    for area in resume_data["areas_for_improvement"]:
                        st.markdown(f"🔍 {area}")
    else:
        st.info("Run the resume ranking process first to see detailed results.")

# Footer
st.markdown("---")
st.caption("AI-Powered Resume Ranking System | Built with Streamlit, LangChain, LangGraph, and Groq")
