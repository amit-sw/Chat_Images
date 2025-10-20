import os
import streamlit as st

import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]="ChatImages"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"

client = ChatOpenAI(model="gpt-5-nano",api_key=os.getenv("OPENAI_API_KEY"))

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Process the uploaded file
    st.write("File uploaded successfully!")
    st.write("File name:", uploaded_file.name)
    mime = uploaded_file.type or "image/png"
    b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    img_json = {
        'type': 'image_url',
        'image_url': data_url
    }
    st.json(img_json)
    st.image(data_url)
    system_prompt = (
        "Analyze the image provided by the user and describe the image. "
        "If it is a plant, tell me if it is healthy or diseased, and what should be done."
    )

    system_msg = SystemMessage(content=system_prompt)

    # Send multimodal content: a short text plus the actual image payload.
    user_msg = HumanMessage(
        content=[
            {"type": "text", "text": "Please analyze this image."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    response = client.invoke([system_msg, user_msg])

    st.write("Response from model:")
    # `response` is a BaseMessage; print only the text content
    st.write(response.content)