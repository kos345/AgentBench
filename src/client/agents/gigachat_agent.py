import os
from ..agent import AgentClient
from gigachat import GigaChat
import argparse
from typing import List

import contextlib
import requests
import warnings
from urllib3.exceptions import InsecureRequestWarning
from src.typings import *
from src.utils import *
import time

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

old_merge_environment_settings = requests.Session.merge_environment_settings
load_dotenv(find_dotenv('.env'))
SYSTEM_PROMPT = ''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/agents/api_agents_gigachat.yaml')
    parser.add_argument('--agent', type=str, default='GigaChat-2')
    return parser.parse_args()

# @contextlib.contextmanager
# def no_ssl_verification():
#     opened_adapters = set()
#
#     def merge_environment_settings(self, url, proxies, stream, verify, cert):
#         # Verification happens only once per connection so we need to close
#         # all the opened adapters once we're done. Otherwise, the effects of
#         # verify=False persist beyond the end of this context manager.
#         opened_adapters.add(self.get_adapter(url))
#
#         settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
#         settings['verify'] = False
#
#         return settings
#
#     requests.Session.merge_environment_settings = merge_environment_settings
#
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore', InsecureRequestWarning)
#             yield
#     finally:
#         requests.Session.merge_environment_settings = old_merge_environment_settings
#
#         for adapter in opened_adapters:
#             try:
#                 adapter.close()
#             except:
#                 pass


class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[Dict[str, Any], None]):
        # check if prompter_name is a method and its variable
        if not prompter:
            return Prompter.default()
        assert isinstance(prompter, dict)
        prompter_name = prompter.get("name", None)
        prompter_args = prompter.get("args", {})
        if hasattr(Prompter, prompter_name) and callable(
            getattr(Prompter, prompter_name)
        ):
            return getattr(Prompter, prompter_name)(**prompter_args)
        return Prompter.default()

    @staticmethod
    def default():
        return Prompter.role_content_dict()

    @staticmethod
    def batched_role_content_dict(*args, **kwargs):
        base = Prompter.role_content_dict(*args, **kwargs)

        def batched(messages):
            result = base(messages)
            return {key: [result[key]] for key in result}

        return batched

    @staticmethod
    def role_content_dict(
        message_key: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
        user_role: str = "user",
        agent_role: str = "agent",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal message_key, role_key, content_key, user_role, agent_role
            role_dict = {
                "user": user_role,
                "agent": agent_role,
            }
            prompt = []
            for item in messages:
                prompt.append(
                    {role_key: role_dict[item["role"]], content_key: item["content"]}
                )
            return {message_key: prompt}

        return prompter

    @staticmethod
    def prompt_string(
        prefix: str = "",
        suffix: str = "AGENT:",
        user_format: str = "USER: {content}\n\n",
        agent_format: str = "AGENT: {content}\n\n",
        prompt_key: str = "prompt",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal prefix, suffix, user_format, agent_format, prompt_key
            prompt = prefix
            for item in messages:
                if item["role"] == "user":
                    prompt += user_format.format(content=item["content"])
                else:
                    prompt += agent_format.format(content=item["content"])
            prompt += suffix
            print(prompt)
            return {prompt_key: prompt}

        return prompter

    @staticmethod
    def claude():
        return Prompter.prompt_string(
            prefix="",
            suffix="Assistant:",
            user_format="Human: {content}\n\n",
            agent_format="Assistant: {content}\n\n",
        )

    @staticmethod
    def palm():
        def prompter(messages):
            return {"instances": [
                Prompter.role_content_dict("messages", "author", "content", "user", "bot")(messages)
            ]}
        return prompter


def check_context_limit(content: str):
    content = content.lower()
    and_words = [
        ["prompt", "context", "tokens"],
        [
            "limit",
            "exceed",
            "max",
            "long",
            "much",
            "many",
            "reach",
            "over",
            "up",
            "beyond",
        ],
    ]
    rule = AndRule(
        [
            OrRule([ContainRule(word) for word in and_words[i]])
            for i in range(len(and_words))
        ]
    )
    return rule.check(content)


class State(TypedDict):
    messages: Annotated[list, add_messages]

class TavilyInput(BaseModel):
    """Входные данные для поискового запроса."""
    query: str = Field(description="Текст поискового запроса")



class GigaAgent(AgentClient):
    def __init__(
        self,
        url,
        credentials,
        proxies=None,
        body=None,
        headers=None,
        return_format="{response}",
        gigachat_api_scope=None,
        prompter=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.credentials = credentials
        self.proxies = proxies or {}
        self.headers = headers or {}
        self.body = body or {}
        self.model = body['model'] or None
        self.return_format = return_format
        self.scope = gigachat_api_scope
        self.prompter = Prompter.get_prompter(prompter)
        if not self.url:
            raise Exception("Please set 'url' parameter")

        self.llm = GigaChat(
            model=body['model'],
            base_url=url,
            credentials=credentials,
            scope=gigachat_api_scope,
            verify_ssl_certs=False,
            profanity_check=False,
            top_p=0
        )


    def _handle_history(self, history: List[dict]) -> Dict[str, Any]:
        return self.prompter(history)

    def chatbot(self, state: State):
        return {"messages": [self.llm_with_tools.invoke([("system", SYSTEM_PROMPT)] + state["messages"])]}

    def get_final_response(self, graph, history, config=None):
        """Использует invoke для получения финального состояния"""
        if history:
            data = {"messages": [(i['role'], i['content']) for i in history]}
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

    def inference(self, history: List[dict]) -> str:
        """
        Основной метод, вызываемый контроллером AgentBench.
        history: список сообщений, где последнее сообщение от 'user' содержит
                 описание задачи и доступные инструменты (в поле 'function').
        """
        # 1. Извлекаем последнее системное сообщение (инструкцию и функции)
        current_state = history[-1]
        if current_state["role"] != "user":
            return "Error: Last message is not from user."

        user_content = current_state.get("content", "")
        available_functions = current_state.get("function", [])  # Список описаний инструментов

        # 2. Подготавливаем промпт для GigaChat, включая описание функций
        llm_messages = [{"role": "user", "content": user_content}]

        # 3. Важно: связываем модель с конкретными инструментами для вызова
        # Для GigaChat может потребоваться преобразование схемы функций в нужный формат
        # Например, через bind_tools или передачу в system prompt
        if available_functions:
            # Преобразуем схемы функций в формат, понятный GigaChat API
            tools_for_llm = self._convert_to_gigachat_tools(available_functions)
            self.llm_with_tools = self.llm.bind_tools(tools_for_llm)
        else:
            self.llm_with_tools = self.llm

        # 4. Вызов модели
        try:
            response = self.llm_with_tools.invoke(llm_messages)

            # 5. Извлекаем ответ модели. Ожидается вызов функции или текстовый ответ.
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Если модель решила вызвать инструмент
                tool_call = response.tool_calls[0]
                result = {
                    "function": tool_call['name'],
                    "arguments": tool_call['args']
                }
                return json.dumps(result)  # Контроллер ожидает JSON-строку
            else:
                # Если модель отвечает текстом (например, "finish")
                return response.content

        except Exception as e:
            return json.dumps({"error": f"Agent inference failed: {str(e)}"})

    def _convert_to_gigachat_tools(self, functions: list) -> list:
        """Конвертирует схемы функций AgentBench в формат GigaChat."""
        converted = []
        for func in functions:
            converted.append({
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {})
                }
            })
        return converted

    # def inference(self, history: List[dict]) -> str:
    #     for _ in range(3):
    #         # try:
    #         memory = MemorySaver()
    #         graph_builder = StateGraph(State)
    #         tool = TavilySearchResults(
    #             max_results=5,
    #             args_schema=TavilyInput
    #         )
    #         tools = [tool]
    #         self.llm_with_tools = self.llm.bind_tools(tools)
    #         graph_builder.add_node("chatbot", self.chatbot)
    #         tool_node = ToolNode(tools=tools)
    #         graph_builder.add_node("tools", tool_node)
    #         graph_builder.add_conditional_edges(
    #             "chatbot",
    #             tools_condition,
    #         )
    #         graph_builder.add_edge("tools", "chatbot")
    #         graph_builder.add_edge(START, "chatbot")
    #         graph = graph_builder.compile(checkpointer=memory)
    #         config2 = {"configurable": {"thread_id": str(uuid4())}}
    #         resp = self.get_final_response(graph, history, config=config2)
    #         return resp
    #         # except AgentClientException as e:
    #         #     raise e
    #         # except Exception as e:
    #         #     print("Warning: ", e)
    #         #     pass
    #         time.sleep(_ + 2)
    #     raise Exception("Failed.")