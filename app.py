import os
import streamlit as st
from chat_ui import start_chat

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]="ChatImages"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"


st.title("Chat with Images")
start_chat()