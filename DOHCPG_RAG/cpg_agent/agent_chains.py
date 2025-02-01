import os
from pathlib import Path
import logging
from time import perf_counter
from dotenv import load_dotenv
from .helper_utils import load_sys_prompts
from .data_models import RouteQuery

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, Runnable

from typing import Tuple, Union, Annotated


# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# class AgentBuilderChain:
    
#     logger = logging.getLogger(__name__)
    
#     def __init__(self, sys_prompts_path: Annotated[Union[str, Path], "PATH TO SYSTEM PROMPTS YAML FILE"]):
#         start_time = perf_counter()
#         self.logger.info("INITIALIZING AGENT BUILDER CHAIN INSTANCE")
#         self.sys_prompts_path = sys_prompts_path
#         self.sys_prompts = load_sys_prompts(sys_prompts_path)
        
#         if self.sys_prompts is not None:
#             # DEBUG
#             self.logger.debug(
#                 f"SYSTEM PROMPTS LOADED SUCCESSFULLY IN {perf_counter() - start_time:.2f}s"
#             )
        
#         self.logger.info(f"AGENT BUILDER CHAIN INITIALIZATION COMPLETED IN {perf_counter() - start_time:.2f}s")
    
#     @classmethod
#     def build_chains(cls):
#         start_time = perf_counter()
#         cls.logger.info("BUILDING AGENT CHAINS")
        
#         # BUILD AGENTS
#         structured_router_agent, cpg_agent, general_cpg_agent, memory_writer_agent = cls.build_agents()
        
#         # BUILD SYSTEM PROMPTS
#         router_prompt, cpg_prompt, general_cpg_prompt, memory_writer_prompt = cls.build_system_prompts()
        
#         # BUILD ROUTER AGENT CHAIN
#         router_agent_chain = router_prompt | structured_router_agent
#         print(f"---TYPE OF ROUTER AGENT CHAIN: {type(router_agent_chain)}---")
        
#         # BUILD CPG AGENT CHAIN
#         cpg_agent_chain = cpg_prompt | cpg_agent
#         print(f"---TYPE OF CPG AGENT CHAIN: {type(cpg_agent_chain)}---")
        
#         # BUILD GENERAL CPG AGENT CHAIN
#         general_cpg_agent_chain = general_cpg_prompt | general_cpg_agent
#         print(f"---TYPE OF GENERAL CPG AGENT CHAIN: {type(general_cpg_agent_chain)}---")
        
#         # BUILD MEMORY WRITER AGENT CHAIN
#         memory_writer_agent_chain = memory_writer_prompt | memory_writer_agent
#         print(f"---TYPE OF MEMORY WRITER AGENT CHAIN: {type(memory_writer_agent_chain)}---")
        
#         cls.logger.info(f"AGENT CHAINS BUILT SUCCESSFULLY IN {perf_counter() - start_time:.2f}s")
        
#         return router_agent_chain, cpg_agent_chain, general_cpg_agent_chain, memory_writer_agent_chain
    
#     def build_agents(self) -> Tuple[Runnable, ChatMistralAI, ChatGoogleGenerativeAI, ChatGroq]:
#         self.logger.info("BUILDING AGENT CHAINS")
        
#         # BUILD ROUTER AGENT WITH GROQ
#         router_agent = ChatGroq(
#             api_key=os.getenv("GROQ_API_KEY", ""),
#             model="llama-3.3-70b-versatile",
#             temperature=0.1,
#         )
        
#         # BIND ROUTER AGENT WITH STRUCTURED OUTPUT
#         structured_router_agent = router_agent.with_structured_output(RouteQuery)
        
#         # BUILD CPG AGENT WITH MISTRAL AI
#         cpg_agent = ChatMistralAI(
#             api_key=os.getenv("MISTRAL_API_KEY", ""),
#             model="mistral-large-latest",
#             temperature=0.1
#         )
        
#         # BUILD GENERAL CPG AGENT WITH GOOGLE GENERATIVE AI
#         general_cpg_agent = ChatGoogleGenerativeAI(
#             api_key=os.getenv("GOOGLE_AI_STUDIO", ""),
#             model="gemini-2.0-flash-exp",
#             temperature=0.2
#         )
        
#         # BUILD MEMORY WRITE AGENT WITH GROQ
#         memory_writer_agent = ChatGroq(
#             api_key=os.getenv("GROQ_API_KEY", ""),
#             model="llama-3.3-70b-versatile",
#             temperature=0.3,
#         )
        
#         return structured_router_agent, cpg_agent, general_cpg_agent, memory_writer_agent
    
#     def build_system_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate, PromptTemplate, ChatPromptTemplate]:       
#         if self.sys_prompts is None:
#             sys_prompts = load_sys_prompts(self.sys_prompts_path)
        
#         # BUILD PROMPT FOR THE ROUTER AGENT
#         router_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", sys_prompts.get("cpg_router_agent_sys_prompt"))
#             ]
#         )
        
#         # BUILD PROMPT FOR THE CPG AGENT
#         cpg_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", sys_prompts.get("cpg_generation_sys_prompt"))
#             ]
#         )
        
#         # BUILD PROMPT FOR THE GENERAL CPG AGENT
#         general_cpg_prompt = PromptTemplate.from_template(sys_prompts.get("general_cpg_agent_sys_prompt"))
        
#         # BUILD PROMPT FOR THE MEMORY WRITER AGENT
#         memory_writer_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", sys_prompts.get("memory_writer_agent_sys_prompt"))
#             ]
#         )
        
#         return router_prompt, cpg_prompt, general_cpg_prompt, memory_writer_prompt

class AgentBuilderChain:
    """Orchestrates creation of AI agent chains with different LLM providers."""
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, sys_prompts_path: Annotated[Path, "PATH TO SYSTEM PROMPTS YAML FILE"]):
        self._init_time = perf_counter()
        self.logger.info("INITIALIZING AGENT BUILDER CHAIN")
        
        # CONVERT TO PATH OBJECT AND VALIDATE
        self.sys_prompts_path = sys_prompts_path
        self._validate_prompts_path()
        
        # Load prompts with error handling
        self.sys_prompts = load_sys_prompts(self.sys_prompts_path)
        
        self.logger.info(f"INITIALIZATION COMPLETE IN {perf_counter() - self._init_time:.2f}s")

    def _validate_prompts_path(self):
        """Ensure prompts file exists and is accessible"""
        if not self.sys_prompts_path.exists():
            raise FileNotFoundError(f"SYSTEM PROMPTS FILE NOT FOUND: {self.sys_prompts_path}")
        if not self.sys_prompts_path.is_file():
            raise ValueError(f"PATH IS NOT A FILE: {self.sys_prompts_path}")

    def build_chains(self) -> Tuple[RunnableSequence, RunnableSequence, RunnableSequence, RunnableSequence]:
        """Construct all agent chains with integrated error handling."""
        start_time = perf_counter()
        self.logger.info("BUILDING AGENT CHAINS")

        try:
            # Build components with validation
            agents = self._build_agents()
            prompts = self._build_system_prompts()
            
            # Create chains using zip for parallel processing
            chains = [
                self._create_chain(prompt, agent)
                for prompt, agent in zip(prompts, agents)
            ]
            
            # Return as typed tuple
            return tuple(chains)

        except Exception as e:
            self.logger.error(f"CHAIN BUILD FAILED: {str(e)}")
            raise
        finally:
            self.logger.info(f"CHAIN BUILDING COMPLETED IN {perf_counter() - start_time:.2f}s")

    def _create_chain(self, prompt: Union[ChatPromptTemplate, PromptTemplate], agent: Runnable) -> RunnableSequence:
        """Validate and create a single agent chain."""
        if not isinstance(prompt, (ChatPromptTemplate, PromptTemplate)):
            raise TypeError(f"INVALID PROMPT TYPE: {type(prompt)}")
        if not isinstance(agent, Runnable):
            raise TypeError(f"INVALID AGENT TYPE: {type(agent)}")
            
        chain = prompt | agent
        self.logger.debug(f"CREATED {type(chain).__name__} CHAIN")
        return chain

    def _build_agents(self) -> Tuple[Runnable, Runnable, Runnable, Runnable]:
        """Initialize LLM agents with API validation."""
        self.logger.info("INITIALIZING LLM AGENTS")
        
        try:
            return (
                self._create_router_agent(),
                self._create_cpg_agent(),
                self._create_general_agent(),
                self._create_memory_agent()
            )
        except KeyError as e:
            raise ValueError(f"MISSING ENVIRONMENT VARIABLE: {str(e)}") from e

    def _get_api_key(self, env_var: str) -> str:
        """Secure API key retrieval with validation."""
        key = os.getenv(env_var)
        if not key:
            raise ValueError(f"MISSING API KEY IN ENVIRONMENT: {env_var}")
        return key

    def _create_router_agent(self) -> Runnable:
        """Create structured output router agent."""
        return ChatGroq(
            api_key=self._get_api_key("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.1
        ).with_structured_output(RouteQuery)

    def _create_cpg_agent(self) -> ChatMistralAI:
        """Create Mistral-based CPG agent."""
        return ChatMistralAI(
            api_key=self._get_api_key("MISTRAL_API_KEY"),
            model="mistral-large-latest",
            temperature=0.1
        )

    def _create_general_agent(self) -> ChatGoogleGenerativeAI:
        """Create Google-based general agent."""
        # return ChatGoogleGenerativeAI(
        #     api_key=self._get_api_key("GOOGLE_AI_STUDIO"),
        #     model="gemini-2.0-flash-exp",
        #     temperature=0.2
        # )
        return ChatGroq(
            api_key=self._get_api_key("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )

    def _create_memory_agent(self) -> ChatGroq:
        """Create memory writer agent."""
        return ChatGroq(
            api_key=self._get_api_key("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )

    def _build_system_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate, PromptTemplate, ChatPromptTemplate]:
        """Construct prompt templates with validation."""
        return (
            ChatPromptTemplate.from_messages([("system", self.sys_prompts["cpg_router_agent_sys_prompt"])]),
            ChatPromptTemplate.from_messages([("system", self.sys_prompts["cpg_generation_sys_prompt"])]),
            PromptTemplate.from_template(self.sys_prompts["general_cpg_agent_sys_prompt"]),
            ChatPromptTemplate.from_messages([("system", self.sys_prompts["memory_writer_agent_sys_prompt"])])
        ) 
 
        
# TEST THE AGENT BUILDER CHAIN
# if __name__ == "__main__":
#     sys_prompts_path = Path("sys_prompts.yaml")
#     agent_builder_chain = AgentBuilderChain(sys_prompts_path)
#     router_chain, cpg_chain, general_chain, memory_chain = agent_builder_chain.build_chains()