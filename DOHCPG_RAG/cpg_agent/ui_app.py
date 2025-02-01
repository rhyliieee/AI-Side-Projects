import streamlit as st
import requests
import json
import time
from typing import Iterator
from uuid import uuid4

st.set_page_config(page_title="CPG Chat Assistant", page_icon="🤖")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False

def stream_response(url: str, headers: dict, payload: dict) -> Iterator[str]:
        
    with requests.post(url, headers=headers, json=payload, stream=True) as response:
        response_lines = response.iter_lines()
        try:
            for line in response_lines:
                if line:
                    decoded_line = str(line.decode("utf-8").removeprefix("data: ").removeprefix("[DONE]"))
                    
                    if "[DONE]" in decoded_line:
                        pass
                    
                    dict_line = json.loads(decoded_line)
                    if isinstance(dict_line, dict):
                        content = dict_line.get('content')
                        if content:
                            print(f"CURRENT TEXT: {content}")
                            yield content if content else ""
                    pass
        except Exception as e:
            exc_word = line.decode("utf-8")
            if "[DONE]" in exc_word:
                pass
        

def main():
    initialize_session_state()
    
    st.title("CPG Chat Assistant")
    
    # API Key input in sidebar
    with st.sidebar:
        st.session_state.api_key = st.text_input("Enter API Key", type="password")
        st.markdown("---")
        if st.session_state.processing:
            st.markdown("🤔 AI is thinking...")
    
    # Chat messages display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask something about CPGs...", disabled=st.session_state.processing):
        # Don't allow empty API key
        if not st.session_state.api_key:
            st.error("Please enter an API key in the sidebar.")
            return      
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add assistant message placeholder
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            try:
                st.session_state.processing = True
                                                
                headers = {
                    "Content-Type": "application/json",
                    "CPG-API-KEY": st.session_state.api_key
                }
                payload = {"messages": [{"userId": "sample-user-id", "content": prompt}], "sessionId": str(uuid4()).replace('-', '_')}
                api_url = "http://127.0.0.1:8008/api/v1/cpg_chat"  # Replace with actual API endpoint
                                
                with st.spinner("Generating response..."):       
                    full_response = st.write_stream(stream_response(api_url, headers, payload))
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                st.rerun() 
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                st.session_state.processing = False
                st.rerun() 

if __name__ == "__main__":
    main()
