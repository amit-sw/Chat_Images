from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage


class AgentState(BaseModel):
    """State of the agent."""
    messages: list = []
    response: str = ""
    category: str = ""

class Category(BaseModel):
    """Category for the agent."""
    category: str
    
def create_llm_msg(system_prompt,history):
    resp=[SystemMessage(content=system_prompt)]
    msgs = history
    for m in msgs:
        if m["role"] == "user":
            resp.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            resp.append(AIMessage(content=m["content"]))
    #print(f"DEBUG CREATE LLM MSGS: {history=}\n{resp=}")
    return resp
    
    
class ChatbotAgent():
    """A chatbot agent that interacts with users."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model_name="gpt-5-nano", openai_api_key=api_key)
        workflow = StateGraph(AgentState)
        workflow.add_node("classifier", self.classifier)
        workflow.add_node("smalltalk_agent", self.smalltalk_agent)
        workflow.add_node("complaint_agent", self.complaint_agent)
        workflow.add_node("status_agent", self.status_agent)
        workflow.add_node("feedback_agent", self.feedback_agent)
        workflow.add_edge(START, "classifier")
        workflow.add_conditional_edges("classifier", self.main_router)
        #workflow.add_edge("classifier", "smalltalk_agent")
        #workflow.add_edge("classifier", "complaint_agent")
        #workflow.add_edge("classifier", "status_agent")
        #workflow.add_edge("classifier", "feedback_agent")
        workflow.add_edge("smalltalk_agent", END)
        workflow.add_edge("complaint_agent", END)
        workflow.add_edge("status_agent", END)
        workflow.add_edge("feedback_agent", END)

        self.graph = workflow.compile()



    def classifier(self, state: AgentState):
        #print("Initial classsifier")
        messages=state.messages
        CLASSIFIER_PROMPT = """
        You are a helpful assistant that classifies user messages into categories.
        Given the following messages, classify them into one of the following categories:
        - smalltalk_agent
        - complaint_agent
        - status_agent
        - feedback_agent

        If you don't know the category, classify it as "smalltalk_agent".
        """
        llm_messages = create_llm_msg(CLASSIFIER_PROMPT, state.messages)
        llm_response = self.model.with_structured_output(Category).invoke(llm_messages)
        category=llm_response.category
        print(f"Classified category: {category}")
        return {"category":category}

    def main_router(self, state: AgentState):
        #print("Routing to appropriate agent based on category")
        #print(f"DEBUG: Current state: {state}")
        #print(f"DEBUG: Current category: {state.category}")
        return state.category

    def smalltalk_agent(self, state: AgentState):
        print("Smalltalk agent processing....")
        SMALLTALK_PROMPT = f"""
        You are a smalltalk agent that engages in casual conversation.
        Given the following messages, respond appropriately to the user's message.
        """
        llm_messages = create_llm_msg(SMALLTALK_PROMPT, state.messages)
        return {"response": self.model.stream(llm_messages), "category": "smalltalk_agent"}

    def complaint_agent(self, state: AgentState):
        print("Complaint agent processing....")
        COMPLAINT_PROMPT = f"""
        You are a complaint agent that addresses user complaints.
        Given the following messages, respond appropriately to the user's complaint.
        """
        llm_messages = create_llm_msg(COMPLAINT_PROMPT, state.messages)
        return {"response": self.model.stream(llm_messages), "category": "complaint_agent"}

    def status_agent(self, state: AgentState):
        print("Status agent processing....")
        STATUS_PROMPT = f"""
        You are a status agent that provides updates on user requests.
        Given the following messages, respond appropriately to the user's request for status.
        """
        llm_messages = create_llm_msg(STATUS_PROMPT, state.messages)
        return {"response": self.model.stream(llm_messages), "category": "status_agent"}

    def feedback_agent(self, state: AgentState):
        print("Feedback agent processing....")
        FEEDBACK_PROMPT = f"""
        You are a feedback agent that collects user feedback.
        Given the following messages, respond appropriately to the user's feedback.
        """
        llm_messages = create_llm_msg(FEEDBACK_PROMPT, state.messages)
        return {"response": self.model.stream(llm_messages), "category": "feedback_agent"}
    
    