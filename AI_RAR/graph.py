from typing import Dict, List, Any, AnyStr
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

# UTILITIES
import time


# INTERNAL IMPORTS
from data_models import MultiJobComparisonState
from utils import CacheManager, flatten
from agents import create_cjc_agent, create_rar_agent

# INITIALIZE CACHE MANAGER
cache_manager = CacheManager()

# NODE TO INITIALIZE STATES
# def initialize_multi_job_state(job_descriptions: List[Dict[str, Any]], 
#                      resumes: List[Dict[str, Any]]) -> MultiJobComparisonState:
#     return {
#         "job_descriptions": job_descriptions,
#         "resumes": resumes,
#         "all_rankings": {},
#         "cross_job_analysis": {},
#         "final_recommendations": None
#     }

# NODE TO ANALYZE AND RANK RESUMES FOR EACH JOB
def rank_resumes_for_jobs(state: MultiJobComparisonState) -> MultiJobComparisonState:
    """RANK RESUMES FOR EACH JOB DESCRIPTION"""
    try:
        # CHECK IF RAR AGENT CHAIN IS ALREADY CACHED
        if not cache_manager.has("rar_agent_chain"):
            cache_manager.set("rar_agent_chain", create_rar_agent())
        
        print(f"---ANALYZING AND RANKING RESUMES IN NODE ONE---")
        
        # GET RAR AGENT CHAIN
        rar_agent_chain = cache_manager.get("rar_agent_chain")

        job_openings = state.get('job_openings', [])
        resumes = state.get('resumes', '')
        processed_job_description = state.get("processed_job_description", [])
        all_rankings = state.get("all_rankings", {})
        processed_jobs = []
        jobs_to_process = job_openings

        # FOR EACH JD, AND RESUME INVOKE RAR AGENT
        for job in jobs_to_process:
            job_name = job.get("name", "")
            job_content = job.get("content", "")

            print(f"---PROCESSING JOB: {job_name}---")

            # Skip if already processed
            if job_name in processed_job_description:
                continue
                
            # TEMPORARY VARIABLE TO HOLD ANALYZED RESUME
            analyzed_resume = []

            for resume in resumes:
                resume_content = resume.page_content
                print(f"---ASSESSING {resume.metadata['source']} AGAINST {job_name} IN NODE ONE")
                
                # INVOKE RAR AGENT
                rar_agent_output = rar_agent_chain.invoke({
                    "job_description": job_content,
                    "resume_content": resume_content
                })
                analyzed_resume.append(rar_agent_output) 
        
            # PERFORM RANKING AND STORE RESULTS
            ranked_resumes = sorted(analyzed_resume, key=lambda record: record.total_score, reverse=True)
            all_rankings[job_name] = ranked_resumes
            processed_jobs.append(job_name)
        
        return {
            "job_openings": job_openings,
            "resumes": resumes,
            "all_rankings": all_rankings,
            "processed_job_description": processed_jobs
        }
    except Exception as e:
        raise RuntimeError(f"ERROR IN 'rank_resumes_for_jobs' NODE: {str(e)}")
    
# NODE TO COMPARE RESUMES ACROSS JOBS
def cross_job_comparison(state: MultiJobComparisonState) -> MultiJobComparisonState:
    """COMPARE RESUMES ACROSS JOBS"""
    try:
        print(f"---PERFORMING CROSS-JOB COMPARISON---")

        job_openings = state.get("job_openings", [])
        resumes = state.get("resumes")
        all_rankings = state.get("all_rankings", "")

        # FLATTEN JOB DESCRIPTION AND CANDIDATE RANKING BEFORE PASSING TO CJC AGENT
        flattened_jd_cr =  flatten(all_rankings, job_openings)

        # CREATE AND CHECK IF CJC AGENT CHAIN IS ALREADY CACHED
        if not cache_manager.has("cjc_agent_chain"):
            cache_manager.set("cjc_agent_chain", create_cjc_agent())
        
        # GET CJC AGENT CHAIN
        cjc_agent_chain = cache_manager.get("cjc_agent_chain")

        print(f"---INVOKING CROSS-JOB COMPARISON AGENT---")

        # INVOKE THE CHAIN
        cjc_result = cjc_agent_chain.invoke({"flattened_jd_cr": flattened_jd_cr})
        
        return {
            "job_openings": job_openings,
            "resumes": resumes,
            "all_rankings": all_rankings,
            "final_recommendations": cjc_result
        }
    except Exception as e:
        max_retries = 3
        retry_delay = 2
            
        for attempt in range(max_retries):
            try:
                
                # GET CJC AGENT CHAIN
                cjc_agent_chain = cache_manager.get("cjc_agent_chain")

                print(f"---INVOKING CROSS-JOB COMPARISON AGENT---")

                # INVOKE THE CHAIN
                cjc_result = cjc_agent_chain.invoke({"flattened_jd_cr": flattened_jd_cr})

                return {
                    "job_openings": job_openings,
                    "resumes": resumes,
                    "all_rankings": all_rankings,
                    "final_recommendations": cjc_result
                }

            except Exception as e:
                if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed, retrying...")
                        time.sleep(retry_delay)
                        continue
                raise e

        raise RuntimeError(f"ERROR IN 'cross_job_comparison' NODE: {str(e)}")

# NODES TO CHECK IF ALL JOBS ARE PROCESSED
def are_all_jobs_processed(state: MultiJobComparisonState) -> str:
    """CHECK IF ALL JOBS HAVE BEEN PROCESSED"""
    try:
        job_openings = state.get("job_openings", [])
        processed_jobs = state.get("processed_job_description", [])
        
        # COUNT UNIQUE JOB NAMES IN job_descriptions
        unique_job_names = set(job.get("name", "") for job in job_openings)
        
        # CHECK IF ALL UNIQUE JOBS HAVE BEEN PROCESSED
        all_processed = len(unique_job_names) == len(set(processed_jobs))
        
        return "CONTINUE" if all_processed else "WAIT"
    except Exception as e:
        raise RuntimeError(f"ERROR IN 'are_all_jobs_processed' NODE: {str(e)}")
        
# CREATE THE MULTI-JOB COMPARISON GRAPH
def create_multi_job_comparison_graph() -> CompiledStateGraph:
    workflow = StateGraph(MultiJobComparisonState)
    
    # ADD NODES
    workflow.add_node("rank_resumes_for_jobs", rank_resumes_for_jobs)
    workflow.add_node("cross_job_comparison", cross_job_comparison)
    
    # ADD EDGES
    workflow.add_edge(START, "rank_resumes_for_jobs")

    # ADD CONDITIONAL EDGE FROM ANALYSIS NODE TO CHECK IF ALL JOBS ARE PROCESSED
    workflow.add_conditional_edges(
            "rank_resumes_for_jobs",
            are_all_jobs_processed,
            {
                "WAIT": "rank_resumes_for_jobs",  # Loop back if not all jobs are processed
                "CONTINUE": "cross_job_comparison"  # Proceed if all jobs are processed
            }
    )

    workflow.add_edge("cross_job_comparison", END)

    print(f"---MULTI-JOB COMPARISON GRAPH COMPILED---")
    
    # COMPILE THE GRAPH
    return workflow.compile()