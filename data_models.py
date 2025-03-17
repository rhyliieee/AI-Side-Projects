from pydantic import BaseModel
from typing import Dict, List, AnyStr

# DATA MODEL FOR RESUME FEEDBACK
class ResumeFeedback(BaseModel):
    analysis: AnyStr
    scores: Dict[AnyStr, int]
    total_score: int
    key_strengths: List[AnyStr]
    areas_for_improvement: List[AnyStr]
