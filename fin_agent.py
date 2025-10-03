import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import streamlit as st
from typing import TypedDict, Annotated, List
import operator
import uuid

# LangChain imports
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor, create_react_agent, tool
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch


# --- Configuration ---
LOCATION = "us-central1"
DB_FILE_PATH = "RAG_data/financial_data.db"
MEMORY_SIZE = 10

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID", "not-set")

table_info = {
     "board_profile_2020": "Contains detailed profiles for all board memebers, including their role, tenure, and profession.",
     "acronyms": "A reference table mapping financial and oil and gas acronyms to their full names. Use this to clarify jargon.",
     "management_companies_2020": "Shareholder returns across time for all companies.",
     "management_pay_2020": "The salaries (if known) of board members and management for all companies ranging from 2017-2020.", 
     "shareholder_investment_2020": "The type of investor (management, retail, institution etc) and their holdings on companies."
}

@tool
def web_research_analyst(query: str) -> str:
    """
    Use this tool for questions about current events, general knowledge, public companies, 
    or any topic that is not covered by the private financial database.
    """
    print(f"--- Web Research Analyst Tool invoked with query: {query} ---")
    # This tool is a wrapper around the Tavily Search API.
    # It requires a TAVILY_API_KEY environment variable to be set.
    tavily_tool = TavilySearch(max_results=3)
    return tavily_tool.invoke(query)

@tool
def sql_research_analyst(query: str) -> str:
    """
    Use this tool to get specific data or answer factual questions from the financial database.
    This tool connects to a SQLite database and can answer questions about board members, financials, and acronyms.
    """
    print(f"--- SQL Research Analyst Tool invoked with query: {query} ---")
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE_PATH}", sample_rows_in_table_info=3)
    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    prompt = hub.pull("hwchase17/react-chat").partial(
        instructions=f"You are an expert SQL data analyst. Your job is to answer the user's question by generating and executing SQL queries against the database." \
        f"You have access to the following database notes from the user: {table_info}"
    )
    agent_runnable = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True, handle_parsing_errors=True)
    
    result = agent_executor.invoke({"input": query, "chat_history": []})
    return result["output"]

@tool
def data_visualization_specialist(query: str):
    """
    Use this tool ONLY when the user asks for a "chart", "plot", "graph", "visualize", or "draw".
    This tool takes a natural language query, generates and executes Python code to create a plot,
    and returns the final matplotlib figure object.
    """
    print(f"--- Data Visualization Specialist Tool invoked with query: {query} ---")
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE_PATH}")
    schema = db.get_table_info()
    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION, temperature=0)
    
    code_generation_prompt = f"""
    You are a data science code generation assistant. Your task is to convert a user's request into a single block of Python code to generate a visualization.
    You have access to the following database schema:
    {schema} and database notes from the user: {table_info}
    The database is located at '{DB_FILE_PATH}'. The code must first connect to the database to get the data, then create a matplotlib figure and assign it to a variable named 'fig'.
    The final line of your code must be `fig` to ensure the figure object is captured. Do NOT use `plt.show()`.
    User Query: "{query}"
    Respond with ONLY the Python code inside a ```python block.
    """
    code_response = llm.invoke(code_generation_prompt)
    generated_code = code_response.content

    if generated_code.startswith("```python"):
        generated_code = generated_code[9:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
    generated_code = generated_code.strip()
    
    print("--- Generated Code ---\n", generated_code)
    
    local_scope = {"pd": pd, "plt": plt, "sns": sns, "sqlite3": sqlite3, "DB_FILE_PATH": DB_FILE_PATH}
    exec(generated_code, globals(), local_scope)
    
    return local_scope.get("fig", "Code executed, but no figure was produced.")


# --- THE MASTER AGENT (The "Team Lead") ---
@st.cache_resource
def get_master_agent():
    print("Setting up Master Agent...")
    
    llm = ChatVertexAI(model_name="gemini-2.5-flash", 
                       project=PROJECT_ID, 
                       location=LOCATION, 
                       temperature=0)
    
    tools = [sql_research_analyst, data_visualization_specialist, web_research_analyst]
    
    prompt = hub.pull("hwchase17/react-chat").partial(
        instructions="""
    You are the lead orchestrator of a team of AI specialists.
    Your job is to understand the user's request and delegate it to the correct specialist tool.
    - For factual answers, use the `sql_research_analyst` tool.
    - For visualizations, use the `data_visualization_specialist` tool.
    - For questions about public information, current events, or general knowledge, use the `web_research_analyst` tool.
    Your final answer must be the direct output from the tool.
    """
    )
    
    agent_runnable = create_react_agent(llm, tools, prompt)
    # memory = ConversationBufferMemory(memory_key="chat_history", 
    #                                   return_messages=True,
    #                                   k=MEMORY_SIZE)
    
    return AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,
        # memory=memory,
        handle_parsing_errors=True,
        return_intermediate_steps=True 
    )

# --- STREAMLIT APP UI ---
st.set_page_config(page_title="Financial Agent", layout="wide")
st.title("🤖 Master Agent")

agent_executor = get_master_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], plt.Figure):
            st.pyplot(message["content"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or request a visualization..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = st.session_state.messages[:-1]
                langchain_history = []
                for msg in history:
                    if msg['role'] == "user":
                        langchain_history.append(HumanMessage(content=msg["content"]))
                    else:
                        content = msg["content"] if not isinstance(msg["content"], plt.Figure) else "A plot was generated."
                        langchain_history.append(AIMessage(content=content))
                # We invoke the agent and get back the intermediate steps
                # result = agent_executor.invoke({"input": prompt})
                result = agent_executor.invoke({"input": prompt,
                                               "chat_history": langchain_history})
                
                # Default to the final text answer
                final_answer = result.get("output", "I encountered an error.")
                
                # Check the intermediate steps for a figure
                if "intermediate_steps" in result and result["intermediate_steps"]:
                    last_step = result["intermediate_steps"][-1]
                    action, observation = last_step
                    
                    # If the last action was our viz tool and it returned a figure, use that!
                    if action.tool == "data_visualization_specialist" and isinstance(observation, plt.Figure):
                        final_answer = observation

                if isinstance(final_answer, plt.Figure):
                    st.pyplot(final_answer)
                else:
                    st.markdown(final_answer)
                
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

                if len(st.session_state.messages) > MEMORY_SIZE:
                    st.session_state.messages = st.session_state.messages[-MEMORY_SIZE:]

            except Exception as e:
                st.error(f"An error occurred: {e}")
