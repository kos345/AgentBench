import argparse
from src.configs import ConfigLoader
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver
import json
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from pydantic_core import PydanticSerializationError
from langchain_community.tools.tavily_search.tool import TavilyInput
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from dotenv import find_dotenv, load_dotenv
import inspect
from uuid import uuid4

load_dotenv(find_dotenv('.env'))
SYSTEM_PROMPT = ''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/agents/api_agents_gigachat.yaml')
    parser.add_argument('--agent', type=str, default='GigaChat-2')
    return parser.parse_args()

class State(TypedDict):
    messages: Annotated[list, add_messages]

class TavilyInput(BaseModel):
    """Входные данные для поискового запроса."""
    query: str = Field(description="Текст поискового запроса")

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke([("system", SYSTEM_PROMPT)] + state["messages"])]}


def stream_graph_updates(graph, user_input, config=None):
    if user_input:
        data = {"messages": ("user", user_input)}
    else:
        data = None

    for event in graph.stream(
            data,
            stream_mode="updates",
            config=config
    ):
        node, data = list(event.items())[0]

        if "messages" in data and len(data['messages']) > 0:
            data["messages"][-1].pretty_print()


def get_final_response(graph, user_input, config=None):
    """Использует invoke для получения финального состояния"""
    if user_input:
        data = {"messages": [("user", user_input)]}
    else:
        data = None

    # Получаем финальное состояние
    final_state = graph.invoke(data, config=config)

    # Ищем последнее сообщение от модели
    messages = final_state.get("messages", [])
    for message in reversed(messages):
        if hasattr(message, 'content') and not isinstance(message, ToolMessage):
            return message.content
    return None


if __name__ == '__main__':
    args = parse_args()
    loader = ConfigLoader()
    config = loader.load_from(args.config)
    agent_config = config[args.agent]['parameters']

    llm = GigaChat(
        model=agent_config['name'],
        base_url=agent_config['url'],
        credentials=agent_config['credentials'],
        scope=agent_config['gigachat_api_scope'],
        verify_ssl_certs=False,
        profanity_check=False,
        top_p=0
    )

    memory = MemorySaver()
    graph_builder = StateGraph(State)
    tool = TavilySearchResults(
        max_results=5,
        args_schema=TavilyInput
    )
    tools = [tool]
    llm_with_tools = llm.bind_tools(tools)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile(checkpointer=memory)


    config2 = {"configurable": {"thread_id": str(uuid4())}}

    while True:
        user_input = input("User:")

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # stream_graph_updates(graph, user_input, config=config2)
        resp = get_final_response(graph, user_input, config=config2)
        print(resp)