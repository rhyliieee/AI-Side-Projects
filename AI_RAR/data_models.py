from pydantic import BaseModel
from typing import Dict, List, AnyStr, Optional, Any, TypedDict, Annotated
from langchain.schema import Document
import operator

# DATA MODEL FOR RAR AGENT
class ResumeFeedback(BaseModel):
    candidate_name: AnyStr
    analysis: AnyStr
    scores: Dict[AnyStr, int]
    total_score: int
    key_strengths: List[AnyStr]
    areas_for_improvement: List[AnyStr]

# DATA MODEL FOR CJC AGENT
class JobResumeMatch(BaseModel):
    job_description_name: str
    candidate_name: str
    match_score: float
    match_explanation: str

class CrossJobMatchResult(BaseModel):
    job_resume_matches: List[JobResumeMatch]
    best_matches_per_job: Dict[str, str]  # job_name -> best_resume_name
    best_matches_per_resume: Dict[str, str]  # resume_name -> best_job_name
    overall_recommendation: str

# OVERALL STATE OF THE MULTI-AGENT SYSTEM
class MultiJobComparisonState(TypedDict):
    job_openings: Annotated[List[Dict[str, Any]], "List of Job Openings"]
    resumes: Annotated[List[Dict[str, Any]], "List of Candidate's Resumes"]
    all_rankings: Annotated[Dict[str, List[ResumeFeedback]], "Ranking of all Candidates per Job Opening"]
    final_recommendations: CrossJobMatchResult
    processed_job_description: Annotated[List[AnyStr], operator.add]

class AnalysisRequest(BaseModel):
    job_openings: List[Dict[AnyStr, Any]]
    resumes: List[Dict[AnyStr, Any]]

class StatusResponse(BaseModel):
    trace_id: str
    status: str
    progress: Optional[Dict[str, str]] = None
    results: Optional[Dict[str, Any]] = None

class StartResponse(BaseModel):
    trace_id: str
    message: str
