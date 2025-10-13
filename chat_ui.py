import streamlit as st

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langsmith import Client
from langchain_core.callbacks import BaseCallbackHandler

from graph import ChatbotAgent

import random
DEBUGGING=1

class LangsmithRunRecorder(BaseCallbackHandler):
    def __init__(self):
        self.root_run_id = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id, **kwargs):
        if parent_run_id is None and self.root_run_id is None:
            self.root_run_id = str(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs):
        if parent_run_id is None and self.root_run_id is None:
            self.root_run_id = str(run_id)

def record_langsmith_feedback(feedback_id: int, feedback_value: str) -> None:
    run_id = st.session_state.get("last_langsmith_run_id")
    if not run_id:
        print("No LangSmith run id available for feedback; skipping logging.")
        return

    try:
        client = Client()
    except Exception as exc:
        print(f"Unable to initialize LangSmith client: {exc}")
        return
    
    source_info = {"source": "streamlit_feedback"}
    thread_id = st.session_state.get("thread_id")
    if thread_id:
        source_info["thread_id"] = thread_id

    try:
        client.create_feedback(
            run_id=run_id,
            key="user_feedback",
            score=feedback_id,
            value=feedback_value,
            source_info=source_info,
        )
        print(f"XXX: created feedback: {run_id=}, {feedback_id=}, {feedback_value=}")
    except Exception as exc:
        print(f"Failed to log LangSmith feedback: {exc}")


def record_feedback():
    print(f"In record_feedback")
    print(f"DEBUG Record Feedback\n\n{st.session_state.get('feedback_id')}")
    feedback_id = st.session_state.get('feedback_id')
    feedback_value = "positive" if feedback_id > 0 else "negative"
    if feedback_id>0:
        st.balloons()
    else:
        st.snow()
    record_langsmith_feedback(feedback_id, feedback_value)
        
def accept_feedback():
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = st.feedback("thumbs", on_change=record_feedback, key="feedback_id")
    if selected is not None:
        st.markdown(f"You selected: {sentiment_mapping[selected]}")
        record_langsmith_feedback(sentiment_mapping[selected])
        
def config_for_langgraph():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = random.randint(1000, 100000000)
    thread_id = st.session_state.thread_id
    
    metadata = {
        "url": st.context.url,
    }

    if st.session_state.get("user_record"):
        user_record = st.session_state.user_record

        if user_record.get("login"):
            metadata["login"] = user_record.get("login")

        if user_record.get("full_name"):
            metadata["full_name"] = user_record.get("full_name")

        if user_record.get("account_name"):
            metadata["account_name"] = user_record.get("account_name")
    
    run_recorder = LangsmithRunRecorder()
    config = {
        "configurable":{"thread_id":thread_id},
        "metadata": metadata,
        "callbacks": [run_recorder],
    }
    return thread_id, config, run_recorder

def start_chat():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
        .header-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Your AI assistant")
    st.markdown("Get instant answers to your questions with AI-powered assistance.")
    st.markdown("<br>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = random.randint(1000, 100000000)
    thread_id = st.session_state.thread_id


    user_record = st.session_state.get('user_record')

    if user_record:
        user_id = user_record.get('id', 0)
        #st.write(f"{user_id=}")
        conv_history = None

        if conv_history:
            for idx, conv in enumerate(conv_history):
                conv_id = conv.get('thread_id')
                short_title = conv.get('short_title')
                conv_name = short_title or f"{conv.get('thread_id')}"
                conv_key = conv_name + f"{idx}"

                if st.sidebar.button(conv_name, type="tertiary", key=conv_key):
                    #st.error("to do")
                    r = conv['conv']
                    #restore_conv_history_to_ui(conv_id, r)


    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                display_text = message["content"].replace("$", "\\$")
                display_text = display_text.replace("\\\\$", "\\$")
                st.markdown(display_text)
                #st.markdown(message["content"].replace("$", "\\$")) 
    
    if prompt := st.chat_input("Ask me anything .", accept_file=True, file_type=["pdf", "md", "doc", "csv","jpg","png"]):
        if prompt and prompt["files"]:
            uploaded_file=prompt["files"][0]
            file_contents, filetype = "XXX XXX XXX TO-DO", "csv"
            if filetype != 'csv':
                prompt.text = prompt.text + f"\n Here are the file contents: {file_contents}"
        
        user_prompt = prompt.text
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        with st.chat_message("user"):
            st.write(user_prompt.replace("$", "\\$"))

        message_history = []
        msgs = st.session_state.messages
    
    # Iterate through chat history, and based on the role (user or assistant) tag it as HumanMessage or AIMessage
        for m in msgs:
            if m["role"] == "user":
                message_history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                message_history.append(AIMessage(content=m["content"]))
        
        app = ChatbotAgent(st.secrets['OPENAI_API_KEY'])
        thread_id, config, run_recorder = config_for_langgraph()
        
        parameters = {'initialMessage': prompt.text, 
                      #'sessionState': st.session_state, 
                        #'sessionHistory': st.session_state.messages, 
                        'message_history': message_history}
        
        if 'csv_data' in st.session_state:
            parameters['csv_data'] = st.session_state['csv_data']
        
        if prompt['files'] and filetype == 'csv':
            parameters['csv_data'] = file_contents
            st.session_state['csv_data'] = file_contents

        with st.spinner("Thinking ...", show_time=True):
            full_response = ""
            




            for s in app.graph.stream(parameters, config):
                if run_recorder.root_run_id and st.session_state.get("last_langsmith_run_id") != run_recorder.root_run_id:
                    st.session_state["last_langsmith_run_id"] = run_recorder.root_run_id
                if DEBUGGING:
                    print(f"GRAPH RUN: {s}")
                for k,v in s.items():
                    if DEBUGGING:
                        print(f"Key: {k}, Value: {v}")
                
                if resp := v.get("responseToUser"):
                    with st.chat_message("assistant"):
                        # Clean up response: remove weird line breaks
                        cleaned_resp = resp.replace('\n', ' ').replace('  ', ' ')
                        st.markdown(cleaned_resp, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": cleaned_resp})
                        accept_feedback()
                        #save_conv_history_to_db(thread_id)
                
                if resp := v.get("response"):
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        for response in resp:
                            full_response = full_response + response.content
                            display_text = full_response.replace("$", "\\$")
                            display_text = display_text.replace("\\\$", "\\$")
                            placeholder.markdown(display_text)
                            #placeholder.markdown(full_response.replace("$", "\\$"))
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    accept_feedback()

if __name__ == '__main__':
    st.set_page_config(page_title="Your AI assistant")
    start_chat()