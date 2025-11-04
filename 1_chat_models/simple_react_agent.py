from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import tool, create_react_agent, AgentExecutor
import datetime

load_dotenv()

@tool
def search_from_duckduckgo(input_text):
    """This fucnction is used to perform an internet search using the duckduckgo free search engine"""
    search = DuckDuckGoSearchRun()
    result = search.invoke(input_text)
    return result

@tool
def get_system_time():
    """This function gets the current system time"""
    current_time = datetime.datetime.now()
    return current_time

llm = ChatOpenAI(model="gpt-4o")

prompt_template = hub.pull("hwchase17/react")

tools = [search_from_duckduckgo, get_system_time]

agent = create_react_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input_text = input("Enter your question here: ")
final_answer = agent_executor.invoke({"input": input_text})

print(final_answer)