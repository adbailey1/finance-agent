import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import streamlit as st
from typing import List
import json

from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_tavily import TavilySearch

from google.oauth2 import service_account
from google.cloud import storage


# --- Configuration ---
LOCATION = "us-central1"
DB_FILE_PATH = "RAG_data/financial_data.db"
MEMORY_SIZE = 10

load_dotenv()

table_info = {
     "board_profile_2020": "Contains detailed profiles for all board memebers, including their role, tenure, and profession.",
     "acronyms": "A reference table mapping financial and oil and gas acronyms to their full names. Use this to clarify jargon.",
     "management_companies_2020": "Shareholder returns across time for all companies.",
     "management_pay_2020": "The salaries (if known) of board members and management for all companies ranging from 2017-2020.", 
     "shareholder_investment_2020": "The type of investor (management, retail, institution etc) and their holdings on companies."
}

st.subheader("Secrets Health Check")
secrets_ok = True
if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets and st.secrets["GCP_SERVICE_ACCOUNT_JSON"]:
    st.success("✅ GCP_SERVICE_ACCOUNT_JSON secret found.")
    try:
        # Test if the JSON is valid
        json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
        st.success("✅ GCP_SERVICE_ACCOUNT_JSON is valid JSON.")
    except (json.JSONDecodeError, TypeError):
        st.error("❌ GCP_SERVICE_ACCOUNT_JSON is NOT valid JSON. Please check your copy-paste.")
        secrets_ok = False
else:
    st.error("❌ GCP_SERVICE_ACCOUNT_JSON secret NOT found or is empty.")
    secrets_ok = False

if "PROJECT_ID" in st.secrets and st.secrets["PROJECT_ID"]:
    st.success("✅ PROJECT_ID secret found.")
else:
    st.warning("⚠️ PROJECT_ID secret not found. Using local .env value (if available).")

# Stop the app if secrets are not configured correctly in the deployed environment
if "STREAMLIT_SERVER_RUNNING_ON" in os.environ and not secrets_ok:
    st.stop()

# @st.cache_resource
# def get_gcp_credentials():
#     """
#     Loads Google Cloud credentials securely. In Streamlit Cloud, it uses a JSON 
#     string from secrets. Locally, it falls back to Application Default Credentials.
#     """
#     # Check if we are in the Streamlit Cloud environment by looking for the specific secret
#     if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
#         print("Loading credentials from Streamlit secrets...")
#         gcp_json_credentials_str = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON")
#         if not gcp_json_credentials_str:
#             st.error("GCP_SERVICE_ACCOUNT_JSON secret is empty!")
#             return None
#         try:
#             credentials_info = json.loads(gcp_json_credentials_str)
#             return service_account.Credentials.from_service_account_info(credentials_info)
#         except json.JSONDecodeError:
#             st.error("Failed to parse GCP_SERVICE_ACCOUNT_JSON. Please ensure it's a valid JSON string.")
#             return None
#     else:
#         # If not on Streamlit (i.e., running locally), we can use the default credentials
#         # set up by the `gcloud auth application-default login` command.
#         print("Using Application Default Credentials for local development.")
#         return None # The library will automatically find local ADC

@st.cache_resource
def get_gcp_credentials():
    """Loads Google Cloud credentials securely."""
    if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
        try:
            credentials_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
            return service_account.Credentials.from_service_account_info(credentials_info)
        except (json.JSONDecodeError, TypeError):
            return None # Fail gracefully
    else:
        return None # Fallback for local ADC
    
# Load the credentials once at the start of the app
GCP_CREDENTIALS = get_gcp_credentials()
PROJECT_ID = st.secrets.get("PROJECT_ID", os.getenv("PROJECT_ID"))

def download_db_from_gcs():
    """
    Downloads the database from a private GCS bucket using secrets.
    """
    gcs_bucket_name = st.secrets.get("GCS_BUCKET_NAME")
    gcs_file_path = st.secrets.get("GCS_FILE_PATH")
    
    # Create the client and download the file
    storage_client = storage.Client(project=PROJECT_ID, credentials=GCP_CREDENTIALS)
    bucket = storage_client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_file_path)
    
    # Ensure the local directory exists
    os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)
    blob.download_to_filename(DB_FILE_PATH)
    print(f"Database downloaded to {DB_FILE_PATH}")

# Check if the app is running on Streamlit Cloud
IS_DEPLOYED = "GCP_SERVICE_ACCOUNT_JSON" in st.secrets
if IS_DEPLOYED and not os.path.exists(DB_FILE_PATH):
    download_db_from_gcs()

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
    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION, temperature=0, credentials=GCP_CREDENTIALS)
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
    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION, temperature=0, credentials=GCP_CREDENTIALS)
    
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
                       temperature=0, 
                       credentials=GCP_CREDENTIALS)
    
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

if "STREAMLIT_SERVER_RUNNING_ON" not in os.environ or secrets_ok:


    # Check if credentials loaded successfully before proceeding
    if IS_DEPLOYED and GCP_CREDENTIALS is None:
        st.error("Failed to load Google Cloud credentials from Streamlit secrets. Please check your configuration.")
        st.stop()

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
