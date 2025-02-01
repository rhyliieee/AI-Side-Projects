from pydantic import BaseModel, Field
from typing import Literal, Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# DATA MODEL FOR ROUTING QUERY
class RouteQuery(BaseModel):
    """Route the user's query to the most relevant datasource."""
    context: Literal['internal_knowledge', 'external_knowledge'] = Field(
        description="Node to route the given user's question. ",
    )

# OVERALL STATE OF THE GRAPH
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document: Annotated[str, "The CPG document to use to base the response on"]