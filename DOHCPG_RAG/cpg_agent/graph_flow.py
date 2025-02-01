from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableSequence, RunnableConfig
from langchain_core.messages import BaseMessage, AIMessage
from typing import Annotated, Sequence, Any

# LOCAL IMPORTS
from .data_models import AgentState
from .data_loader import ContextLoader
from .agent_chains import AgentBuilderChain
from .helper_utils import structure_context

import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from threading import Lock
import logging

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# DEFINE DEPENDENCY INJECTION CONTAINER
@dataclass
class ServiceContainer:
    router_chain: "RunnableSequence"
    cpg_chain: "RunnableSequence"
    general_chain: "RunnableSequence"
    memory_chain: "RunnableSequence"
    app_config: "AppConfig"
    memory_store: "BaseStore"
    memory_saver: "MemorySaver"

@dataclass
class AppConfig:
    nomic_token: str
    hf_token: str
    dataset_name: str = "DOHCPG-DOCS",
    sys_prompts_path: Path = Path("sys_prompts.yaml")
    
    @classmethod
    def from_env(cls):
        return cls(
            nomic_token=os.getenv("DOHCPG_TOKEN"),
            hf_token=os.getenv("HF_TOKEN"),
            dataset_name=os.getenv("DATASET_NAME"),
            sys_prompts_path=Path(os.getenv("SYS_PROMPTS_PATH"))
        )

# SINGLETON INIITIALIZATION
class ServiceInitializer:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        try:
            self.app_config = AppConfig.from_env()
            agent_builder = AgentBuilderChain(sys_prompts_path=self.app_config.sys_prompts_path)
            
            self.router_chain, self.cpg_chain, \
                self.general_chain, self.memory_chain = agent_builder.build_chains()
            
            self.memory_store = InMemoryStore()
            self.memory_saver = MemorySaver()
            self.health_check()
        except Exception as e:
            logging.critical("CRITICAL INITIALIZATION FAILED", exc_info=True)
            raise RuntimeError("Critical service failure") from e
    
    def health_check(self):
        if not all(
            [
                self.router_chain,
                self.cpg_chain,
                self.general_chain,
                self.memory_chain,
                self.memory_store,
                self.memory_saver
            ]
        ):
            raise Exception("Chain initialization incomplete")
    
    @contextmanager
    def get_container(self):
        yield ServiceContainer(
            router_chain=self.router_chain,
            cpg_chain=self.cpg_chain,
            general_chain=self.general_chain,
            memory_chain=self.memory_chain,
            app_config=self.app_config,
            memory_store=self.memory_store,
            memory_saver=self.memory_saver
        )

def router_node(state: AgentState, container: ServiceContainer):
    try:
        logging.debug("ROUTER NODE INVOKED")
        print("---ROUTING THE QUERY---")
        
        # OBTAIN THE MESSAGES LIST
        messages = state["messages"]
        last_message = messages[-1]
        
        # INVOKE THE ROUTER CHAIN
        router_response = container.router_chain.invoke({"$user_query": last_message.content})
        
        # ROUTE TO THE RELEVANT NODE BASED ON ROUTER RESPONSE
        if router_response.context == "internal_knowledge":
            print("---ROUTING TO INTERNAL KNOWLEDGE---")
            return Command(goto="general_agent_node", update={"messages": messages})
        elif router_response.context == "external_knowledge":
            print("---ROUTING TO EXTERNAL KNOWLEDGE---")
            return Command(goto="retriever_node", update={"messages": messages})
    except Exception as e:
        logging.error("ERROR IN 'router_node'", exc_info=True)
        raise f"ROUTER NODE FAILED: {e}"

# DEFINE RETRIEVER NODE
def retriever_node(state: AgentState, container: ServiceContainer):
    try:
        logging.debug("RETRIEVER NODE INVOKED")
        print("---RETRIEVING DOCUMENTS---")
        
        # RETRIEVE THE USER'S QUESTION FROM THE STATE
        question = state["messages"][-1].content
               
        # LOAD THE CONTEXT LOADER
        context_loader = ContextLoader(
            query=question,
            dataset_name=container.app_config.dataset_name,
            nomic_token=container.app_config.nomic_token,
            hf_token=container.app_config.hf_token
        )
        
        # RETRIEVE TABLE DATA AND INDEX FROM THE CONTEXT LOADER
        text_data, top_indices = context_loader.pull_context()
        
        # CONVERT TEXT DATA TO PYARROW TABLE
        text_data = text_data.tb
        
        # FORMAT THE DATA FOR THE AGENT TO USE
        structured_context = structure_context(text_data, top_indices)
        
        with open("structured_context.txt", "w") as file:
            file.write(structured_context)
        
        print("---CONTEXT RETRIEVED---")
        
        return {"document": structured_context, "messages": state["messages"]}
    
    except Exception as e:
        logging.error("ERROR IN 'retriever_node'", exc_info=True)
        raise f"RETRIEVER NODE FAILED: {e}"

def general_agent_node(state: AgentState, container: ServiceContainer, config: RunnableConfig):
    try:
        logging.debug("GENERAL AGENT NODE INVOKED")
        print("---GENERAL AGENT NODE INVOKED---")
        
        # GET USER ID FROM CONFIG
        user_id = config.get("configurable", {}).get("user_id")
        print(f"---USER ID: {user_id}---") # USERID IS NONE HERE
        print(f"---MEMORY STORE: {type(container.memory_store)}---")
                
        # RETRIEVE MEMORY FROM THE STORE
        namespace = ("memory", user_id)
        key = "user_memory"
        existing_memory = container.memory_store.get(namespace, key)
        
        print("---GENERAL CPG AGENT GENERATING RESPONSE---")
        messages = state["messages"] # OBTAIN THE CURRENT LIST OF MESSAGES
        last_message = messages[-1] # OBTAIN THE LAST MESSAGE
        
        # EXTRACT ACTUAL MEMORY IF IT EXISTS AND ADD A PREFIX
        if existing_memory:
            logging.debug("EXISTING MEMORY FOUND")
            existing_memory_content = existing_memory.value.get("memory")
        else:
            existing_memory_content = "No existing memory found." if not messages else messages
        
        agent_response = container.general_chain.invoke({"$user_memory": existing_memory_content, "$user_query": last_message.content}) # invoke the agent model
        
        return {"messages": [agent_response]}
    
    except Exception as e:
        logging.error("ERROR IN 'general_agent_node'", exc_info=True)
        raise f"GENERAL AGENT NODE FAILED: {e}"

def cpg_agent_node(state: AgentState, container: ServiceContainer, config: RunnableConfig):
    try:
        logging.debug("CPG AGENT NODE INVOKED")
        # GET USER ID FROM CONFIG
        user_id = config.get("configurable").get("user_id")
        
        # RETRIEVE MEMORY FROM THE STORE
        namespace = ("memory", user_id)
        key = "user_memory"
        existing_memory = container.memory_store.get(namespace, key)
        
        logging.debug("---CPG AGENT GENERATING RESPONSE---")
        messages = state["messages"] # OBTAIN THE CURRENT LIST OF MESSAGES
        last_message = messages[-1] # OBTAIN THE LAST MESSAGE
        docs = state["document"] # OBTAIN THE DOCUMENTS FROM THE STATE
        
        # EXTRACT ACTUAL MEMORY IF IT EXISTS AND ADD A PREFIX
        if existing_memory:
            logging.debug("EXISTING MEMORY FOUND")
            existing_memory_content = existing_memory.value.get("memory")
        else:
            existing_memory_content = "No existing memory found."
        
        # GENERATE THE RESPONSE USING THE DOCUMENTS AND THE USER'S QUESTION
        cpg_agent_response = container.cpg_chain.invoke({"$user_query": last_message.content, "$user_memory": existing_memory_content, "$document": docs})
        
        # RETURN AND UPDATE THE STATE
        return {"messages": [cpg_agent_response]}
    
    except Exception as e:
        logging.error("ERROR IN 'cpg_agent_node'", exc_info=True)
        raise f"CPG AGENT NODE FAILED: {e}"

def write_memory_node(state: AgentState, container: ServiceContainer, config: RunnableConfig):
    try:
        logging.debug("WRITE MEMORY NODE INVOKED")
        # GENERATE RESPONSE FROM THE MEMORY WRITER AGENT
        messages = state["messages"] # OBTAIN THE CURRENT LIST OF MESSAGES
                
        # IF MESSAGES IS MORE THAN 3, THEN SUMMARIZE THE MESSAGE
        if len(messages) > 3:
            print("---MEMORY WRITER AGENT GENERATING RESPONSE---")
            # Get the user ID from the config
            user_id = config.get("configurable").get("user_id")

            # Retrieve existing memory from the store
            namespace = ("memory", user_id)
            key = "user_memory"
            existing_memory = container.memory_store.get(namespace, "user_memory")
                
            # Extract the memory
            if existing_memory:
                logging.debug("EXISTING MEMORY FOUND")
                existing_memory_content = existing_memory.value.get('memory')
            else:
                existing_memory_content = "No existing memory found."
            
            # PARSE THE MESSAGES FROM THE MESSAGES LIST
            messages_history = [message.content for message in messages]
                
            new_memory = container.memory_chain.invoke(input={"$memory": existing_memory_content, "$chat_history": "\n".join(messages_history)})
                        
            # WRITE A VALUE AS A DICT THEN STORE IN THE MEMORY
            container.memory_store.put(namespace, key, {"memory": new_memory.content})
            
    except Exception as e:
        logging.error("ERROR IN 'write_memory_node'", exc_info=True)
        raise f"WRITE MEMORY NODE FAILED: {e}"

def graph_builder() -> CompiledStateGraph:
        initializer = ServiceInitializer()
        
        with initializer.get_container() as container:
            cpg_workflow = StateGraph(AgentState)
            
            # ADD NODES TO THE GRAPH
            cpg_workflow.add_node("router_node", lambda state: router_node(state, container))
            cpg_workflow.add_node("retriever_node", lambda state: retriever_node(state, container))
            cpg_workflow.add_node("general_agent_node", lambda state, config: general_agent_node(state, container, config))
            cpg_workflow.add_node("cpg_agent_node", lambda state, config: cpg_agent_node(state, container, config))
            cpg_workflow.add_node("write_memory_node", lambda state, config: write_memory_node(state, container, config))
            
            # CONNECT NODES WITH EDGES
            cpg_workflow.add_edge(START, "router_node") # START TO ROUTER NODE
            cpg_workflow.add_edge("retriever_node", "cpg_agent_node")
            cpg_workflow.add_edge("general_agent_node", "write_memory_node")
            cpg_workflow.add_edge("cpg_agent_node", "write_memory_node")
            cpg_workflow.add_edge("write_memory_node", END)
            
            # COMPILE THE GRAPH
            cpg_graph = cpg_workflow.compile(
                checkpointer=container.memory_saver, # MEMORY SAVER FOR CHECKPOINTING
                store=container.memory_store, # MEMORY STORE FOR STORING STATE
            )
            return cpg_graph

# # TEST THE CPG GRAPH
# if __name__ == "__main__":
#     from langchain_core.messages import HumanMessage, AIMessageChunk
#     cpg_graph = graph_builder()
#     # TEST THE CPG GRAPH
#     config = {"configurable": {"thread_id": "1", "user_id": "1"}}

#     # USER INPUT
#     input_message = HumanMessage(content="Wow! That's very informative. Thanks.")
#     first = True
#     # INVOKE THE GRAPH
#     # Stream LLM tokens for messages generated in nodes
#     async for msg, metadata in cpg_graph.astream(
#         {"messages": [input_message]}, 
#         config=config, 
#         stream_mode="messages"
#     ):
#         if metadata.get("langgraph_node") in ['general_agent_node', 'cpg_agent_node']:
#             if isinstance(msg, AIMessageChunk):
#                 if first:
#                     gathered = msg
#                     first = False
#                 else:
#                     gathered += msg
                
#                 print(msg.content, end='')
