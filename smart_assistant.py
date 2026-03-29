from __future__ import annotations

import argparse
import os
import re
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, ValidationError

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


class RequestType(str, Enum):
    QUESTION = "question"
    TASK = "task"
    SMALL_TALK = "small_talk"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class Classification(BaseModel):
    request_type: RequestType = Field(description="Тип запроса пользователя")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность классификатора")
    reasoning: str = Field(min_length=1, description="Краткое обоснование решения")


class AssistantResponse(BaseModel):
    content: str = Field(min_length=1, description="Текст ответа ассистента")
    request_type: RequestType = Field(description="Определённый тип запроса")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность классификатора")
    tokens_used: int = Field(ge=0, description="Оценка использованных токенов")


MemoryStrategy = Literal["buffer", "summary"]


CHARACTER_PROMPTS: Dict[str, str] = {
    "friendly": (
        "Ты умный дружелюбный ассистент для внутренней разработки. "
        "Пишешь тепло, уверенно, без воды. Допустимы лёгкие эмодзи, но умеренно. "
        "Запоминай факты о пользователе и опирайся на контекст."
    ),
    "professional": (
        "Ты профессиональный ассистент для инженерной среды. "
        "Стиль точный, сдержанный, спокойный. Без фамильярности, без лишней эмоциональности. "
        "При неоднозначности выбирай наиболее практичный и безопасный вариант."
    ),
    "sarcastic": (
        "Ты умный ассистент с лёгкой иронией. "
        "Сарказм мягкий, без токсичности, без пассивной агрессии. "
        "Главное — полезность, ясность и точность. Если пользователь уже сообщил факт, "
        "напоминай его с оттенком самоиронии."
    ),
    "pirate": (
        "Ты ассистент-пират. Говоришь колоритно, но читаемо и по делу. "
        "Иногда используешь 'Арр', 'матрос', 'тысяча чертей', но не превращай ответ в пародию. "
        "Даже в пиратском стиле сохраняй инженерную точность."
    ),
}


REQUEST_TYPE_STYLES: Dict[RequestType, str] = {
    RequestType.QUESTION: "cyan",
    RequestType.TASK: "green",
    RequestType.SMALL_TALK: "magenta",
    RequestType.COMPLAINT: "yellow",
    RequestType.UNKNOWN: "red",
}


class MemoryManager:
    def __init__(
        self,
        strategy: MemoryStrategy = "buffer",
        max_messages: int = 20,
        summary_trigger: int = 24,
        summary_keep_last: int = 8,
        summarizer: Optional[Runnable[Any, Any]] = None,
    ) -> None:
        self.strategy: MemoryStrategy = strategy
        self.max_messages: int = max_messages
        self.summary_trigger: int = summary_trigger
        self.summary_keep_last: int = summary_keep_last
        self.summarizer: Optional[Runnable[Any, Any]] = summarizer

        self._messages: List[BaseMessage] = []
        self._summary: str = ""
        self._entities: Dict[str, str] = {}

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def entities(self) -> Dict[str, str]:
        return dict(self._entities)

    def set_strategy(self, strategy: MemoryStrategy) -> None:
        self.strategy = strategy

    def set_summarizer(self, summarizer: Runnable[Any, Any]) -> None:
        self.summarizer = summarizer

    def clear_history(self) -> None:
        self._messages.clear()
        self._summary = ""

    def clear_all(self) -> None:
        self.clear_history()
        self._entities.clear()

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))
        self._extract_entities(content)

    def add_ai_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))
        self._trim_or_summarize()

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.add_user_message(user_text)
        self.add_ai_message(assistant_text)

    def get_prompt_history(self) -> List[BaseMessage]:
        if self.strategy == "summary":
            history: List[BaseMessage] = []
            if self._summary.strip():
                history.append(
                    SystemMessage(
                        content=(
                            "Сжатый контекст предыдущего диалога. "
                            "Считай его достоверной памятью:\n"
                            f"{self._summary.strip()}"
                        )
                    )
                )
            history.extend(self._messages[-self.summary_keep_last :])
            return history

        return self._messages[-self.max_messages :]

    def get_entity_memory_as_text(self) -> str:
        if not self._entities:
            return "Нет сохранённых пользовательских фактов."

        rows: List[str] = []
        for key, value in sorted(self._entities.items()):
            rows.append(f"- {key}: {value}")
        return "\n".join(rows)

    def _trim_or_summarize(self) -> None:
        match self.strategy:
            case "buffer":
                if len(self._messages) > self.max_messages:
                    self._messages = self._messages[-self.max_messages :]
            case "summary":
                self._maybe_refresh_summary()
            case _:
                self._messages = self._messages[-self.max_messages :]

    def _maybe_refresh_summary(self) -> None:
        if len(self._messages) <= self.summary_trigger:
            return

        if self.summarizer is None:
            self._messages = self._messages[-self.max_messages :]
            return

        split_index: int = max(0, len(self._messages) - self.summary_keep_last)
        compress_chunk: List[BaseMessage] = self._messages[:split_index]
        tail_chunk: List[BaseMessage] = self._messages[split_index:]

        if not compress_chunk:
            return

        serialized_chunk: str = self._serialize_messages(compress_chunk)
        summary_prompt: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "Ты memory-manager. Сожми историю диалога в плотное summary для дальнейшего использования в prompt. "
                    "Сохраняй: имена, предпочтения, задачи, обещания, ограничения, принятые решения, тон общения. "
                    "Не добавляй догадки. Пиши компактно, списком фактов."
                )
            ),
            HumanMessage(
                content=(
                    f"Текущее summary:\n{self._summary or 'Пусто'}\n\n"
                    f"Новый фрагмент диалога:\n{serialized_chunk}\n\n"
                    "Верни обновлённое summary."
                )
            ),
        ]

        try:
            raw_result: Any = self.summarizer.invoke(summary_prompt)
            summary_text: str = self._extract_text(raw_result).strip()
            if summary_text:
                self._summary = summary_text
                self._messages = tail_chunk
            else:
                self._messages = self._messages[-self.max_messages :]
        except Exception:
            self._messages = self._messages[-self.max_messages :]

    def _serialize_messages(self, messages: Sequence[BaseMessage]) -> str:
        rows: List[str] = []
        for message in messages:
            role: str = message.type.upper()
            rows.append(f"{role}: {self._extract_text(message)}")
        return "\n".join(rows)

    def _extract_entities(self, content: str) -> None:
        patterns: List[Tuple[str, str]] = [
            (r"\bменя зовут\s+([A-ZА-ЯЁ][a-zа-яё-]+)\b", "name"),
            (r"\bмо[йя]\s+любим(?:ый|ая)\s+язык\s*[—:-]?\s*([A-Za-zА-Яа-яЁё0-9_+\-#]+)", "favorite_language"),
            (r"\bя\s+из\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\- ]+)", "city_or_origin"),
            (r"\bя\s+живу\s+в\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\- ]+)", "city"),
            (r"\bмой\s+любимый\s+фреймворк\s*[—:-]?\s*([A-Za-zА-Яа-яЁё0-9_+\-#\.]+)", "favorite_framework"),
        ]

        lowered: str = content.strip()
        for pattern, key in patterns:
            match_obj: Optional[re.Match[str]] = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match_obj:
                self._entities[key] = match_obj.group(1).strip()

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, BaseMessage):
            return str(payload.content)
        if isinstance(payload, str):
            return payload
        if hasattr(payload, "content"):
            return str(payload.content)
        return str(payload)


class BaseAssistant(ABC):
    def __init__(
        self,
        *,
        model_name: str,
        fallback_model_name: Optional[str],
        temperature: float,
        character: str,
        memory_strategy: MemoryStrategy,
        console: Optional[Console] = None,
    ) -> None:
        self.console: Console = console or Console()
        self.character: str = character
        self.temperature: float = temperature
        self.model_name: str = model_name
        self.fallback_model_name: Optional[str] = fallback_model_name

        set_llm_cache(InMemoryCache())

        self.primary_model: ChatOpenAI = self._build_chat_model(model_name=model_name, temperature=temperature)
        self.fallback_model: Optional[ChatOpenAI] = (
            self._build_chat_model(model_name=fallback_model_name, temperature=temperature)
            if fallback_model_name
            else None
        )
        self.resilient_model: Runnable[Any, Any] = self._wrap_with_fallbacks(self.primary_model, self.fallback_model)

        self.memory: MemoryManager = MemoryManager(strategy=memory_strategy)
        self.memory.set_summarizer(self.resilient_model)

        self._build_runtime()

    def _build_chat_model(self, model_name: Optional[str], temperature: float):
        if not model_name:
            raise ValueError("model_name cannot be empty")

        return ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def _wrap_with_fallbacks(
        self,
        primary: Runnable[Any, Any],
        fallback: Optional[Runnable[Any, Any]],
    ) -> Runnable[Any, Any]:
        if fallback is None:
            return primary
        return primary.with_fallbacks([fallback])

    @abstractmethod
    def _build_runtime(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def process(self, query: str) -> AssistantResponse:
        raise NotImplementedError

    def set_character(self, character: str) -> None:
        if character not in CHARACTER_PROMPTS:
            raise ValueError(f"Unsupported character: {character}")
        self.character = character
        self._build_runtime()

    def set_memory_strategy(self, strategy: MemoryStrategy) -> None:
        self.memory.set_strategy(strategy)

    def clear_history(self) -> None:
        self.memory.clear_history()

    def status_snapshot(self) -> Dict[str, Union[str, int]]:
        return {
            "character": self.character,
            "memory_strategy": self.memory.strategy,
            "history_messages": self.memory.message_count,
            "entity_facts": len(self.memory.entities),
            "summary_exists": "yes" if bool(self.memory.summary.strip()) else "no",
            "model": self.model_name,
            "fallback_model": self.fallback_model_name or "disabled",
        }


class SmartAssistant(BaseAssistant):
    def _build_runtime(self) -> None:
        self.classification_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Classification)
        self.classifier_chain: Runnable[Any, Classification] = self._build_classifier_chain()
        self.handler_chains: Dict[RequestType, Runnable[Any, str]] = self._build_handler_chains()
        self.router: Runnable[Any, str] = self._build_router()

    def _build_classifier_chain(self) -> Runnable[Any, Classification]:
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Ты классификатор пользовательских запросов для CLI-ассистента.\n"
                        "Определи один тип из списка:\n"
                        "- question: пользователь хочет получить объяснение, факт, анализ, ответ на вопрос.\n"
                        "- task: пользователь просит что-то создать, написать, придумать, сделать.\n"
                        "- small_talk: приветствие, знакомство, лёгкая беседа, обмен репликами.\n"
                        "- complaint: недовольство, жалоба, раздражение, претензия к работе.\n"
                        "- unknown: бессмысленный, слишком шумный или нераспознаваемый ввод.\n\n"
                        "Примеры:\n"
                        "Запрос: Привет!\n"
                        'Ответ: {"request_type":"small_talk","confidence":0.98,"reasoning":"Обычное приветствие"}\n\n'
                        "Запрос: Что такое LCEL?\n"
                        'Ответ: {"request_type":"question","confidence":0.95,"reasoning":"Пользователь просит объяснение термина"}\n\n'
                        "Запрос: Напиши короткий стих про Python\n"
                        'Ответ: {"request_type":"task","confidence":0.96,"reasoning":"Пользователь просит сгенерировать контент"}\n\n'
                        "Запрос: Это ужасно работает, почему так медленно?\n"
                        'Ответ: {"request_type":"complaint","confidence":0.93,"reasoning":"Есть жалоба на качество работы"}\n\n'
                        "Запрос: asdf qwerty ыфвфыв\n"
                        'Ответ: {"request_type":"unknown","confidence":0.72,"reasoning":"Смысл запроса не определяется"}\n\n'
                        "{format_instructions}"
                    ),
                ),
                ("human", "Классифицируй запрос:\n{query}"),
            ]
        )

        return (
            {
                "query": RunnablePassthrough(),
                "format_instructions": RunnableLambda(lambda _: self.classification_parser.get_format_instructions()),
            }
            | prompt
            | self.resilient_model
            | self.classification_parser
        )

    def _build_handler_chains(self) -> Dict[RequestType, Runnable[Any, str]]:
        return {
            RequestType.QUESTION: self._build_handler_chain(
                request_type=RequestType.QUESTION,
                domain_prompt=(
                    "Это информационный запрос. Дай точный, полезный, структурный ответ. "
                    "Если данных недостаточно — честно скажи об этом, но всё равно предложи лучший следующий шаг."
                ),
            ),
            RequestType.TASK: self._build_handler_chain(
                request_type=RequestType.TASK,
                domain_prompt=(
                    "Это запрос на выполнение задачи. Сделай результат качественно и полностью. "
                    "Если нужна структура — дай её сразу. Если можно выдать готовый артефакт в тексте — выдай."
                ),
            ),
            RequestType.SMALL_TALK: self._build_handler_chain(
                request_type=RequestType.SMALL_TALK,
                domain_prompt=(
                    "Это small talk. Поддержи беседу естественно. "
                    "Если пользователь представился или сообщил личный факт — отрази, что ты это запомнил."
                ),
            ),
            RequestType.COMPLAINT: self._build_handler_chain(
                request_type=RequestType.COMPLAINT,
                domain_prompt=(
                    "Это жалоба. Сначала прояви эмпатию, затем кратко проясни суть проблемы и предложи практичное решение."
                ),
            ),
            RequestType.UNKNOWN: self._build_handler_chain(
                request_type=RequestType.UNKNOWN,
                domain_prompt=(
                    "Запрос неясен. Вежливо сообщи, что ты не уверен в интерпретации, и предложи 2-4 варианта того, "
                    "как пользователь может переформулировать мысль."
                ),
            ),
        }

    def _build_handler_chain(self, request_type: RequestType, domain_prompt: str) -> Runnable[Any, str]:
        system_prompt: str = (
            f"{CHARACTER_PROMPTS[self.character]}\n\n"
            f"Текущая ветка обработки: {request_type.value}.\n"
            f"{domain_prompt}\n\n"
            "Используй память диалога и сохранённые факты о пользователе. "
            "Не выдумывай факты, которых нет в истории или entity-memory."
        )

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "system",
                    "Сохранённые пользовательские факты:\n{entity_memory}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
            ]
        )

        return prompt | self.resilient_model | StrOutputParser()

    def _build_router(self) -> Runnable[Any, str]:
        return RunnableBranch(
            (lambda payload: payload["classification"].request_type == RequestType.QUESTION, self.handler_chains[RequestType.QUESTION]),
            (lambda payload: payload["classification"].request_type == RequestType.TASK, self.handler_chains[RequestType.TASK]),
            (lambda payload: payload["classification"].request_type == RequestType.SMALL_TALK, self.handler_chains[RequestType.SMALL_TALK]),
            (lambda payload: payload["classification"].request_type == RequestType.COMPLAINT, self.handler_chains[RequestType.COMPLAINT]),
            self.handler_chains[RequestType.UNKNOWN],
        )

    def classify(self, query: str) -> Classification:
        try:
            result: Classification = self.classifier_chain.invoke(query)
            return result
        except (ValidationError, ValueError, TypeError):
            return Classification(
                request_type=RequestType.UNKNOWN,
                confidence=0.50,
                reasoning="Ошибка парсинга ответа модели",
            )
        except Exception:
            return Classification(
                request_type=RequestType.UNKNOWN,
                confidence=0.50,
                reasoning="Классификатор недоступен, использован fallback UNKNOWN",
            )

    def process(self, query: str) -> AssistantResponse:
        with self.console.status("[bold blue]Классификация и подготовка ответа...[/bold blue]", spinner="dots"):
            classification: Classification = self.classify(query)
            payload: Dict[str, Any] = {
                "query": query,
                "classification": classification,
                "history": self.memory.get_prompt_history(),
                "entity_memory": self.memory.get_entity_memory_as_text(),
            }

        content: str = self._stream_response(payload, classification.request_type).strip()
        final_content: str = content or "Не удалось сформировать ответ."

        self.memory.add_turn(query, final_content)

        tokens_used: int = self._estimate_tokens(query=query, response=final_content, payload=payload)

        return AssistantResponse(
            content=final_content,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=tokens_used,
        )

    def _stream_response(self, payload: Dict[str, Any], request_type: RequestType) -> str:
        style: str = REQUEST_TYPE_STYLES[request_type]
        title: str = f"[{request_type.value}]"

        chunks: List[str] = []
        rendered_text: Text = Text("", style=style)

        with Live(
            Panel(rendered_text, title=title, border_style=style),
            console=self.console,
            refresh_per_second=24,
            transient=False,
        ) as live:
            try:
                for chunk in self.router.stream(payload):
                    part: str = self._normalize_chunk(chunk)
                    if not part:
                        continue
                    chunks.append(part)
                    rendered_text = Text("".join(chunks), style=style)
                    live.update(Panel(rendered_text, title=title, border_style=style))
            except Exception as exc:
                fallback_text: str = f"Не удалось обработать запрос: {exc}"
                chunks = [fallback_text]
                rendered_text = Text(fallback_text, style="red")
                live.update(Panel(rendered_text, title="[error]", border_style="red"))

        self.console.print()
        return "".join(chunks)

    def _estimate_tokens(self, query: str, response: str, payload: Dict[str, Any]) -> int:
        prompt_parts: List[str] = [query, response, payload.get("entity_memory", "")]
        for message in payload.get("history", []):
            if isinstance(message, BaseMessage):
                prompt_parts.append(str(message.content))

        joined: str = "\n".join(part for part in prompt_parts if part)
        try:
            return int(self.primary_model.get_num_tokens(joined))
        except Exception:
            rough_words: int = len(re.findall(r"\S+", joined))
            return max(1, int(rough_words * 1.35))

    @staticmethod
    def _normalize_chunk(chunk: Any) -> str:
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, BaseMessage):
            return str(chunk.content)
        if hasattr(chunk, "content"):
            return str(chunk.content)
        return str(chunk)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Assistant CLI")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--fallback-model", default=os.getenv("OPENAI_FALLBACK_MODEL", "qwen3-32b"))
    parser.add_argument(
        "--character",
        default="friendly",
        choices=sorted(CHARACTER_PROMPTS.keys()),
    )
    parser.add_argument(
        "--memory",
        default="buffer",
        choices=["buffer", "summary"],
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    return parser


def render_banner(console: Console, assistant: SmartAssistant) -> None:
    snapshot: Dict[str, Union[str, int]] = assistant.status_snapshot()
    console.print(
        Panel(
            Text(
                "Умный ассистент с характером\n"
                f"Характер: {snapshot['character']} | Память: {snapshot['memory_strategy']} | "
                f"Модель: {snapshot['model']}",
                justify="left",
            ),
            border_style="blue",
        )
    )


def render_help(console: Console) -> None:
    table = Table(title="Команды", show_header=True, header_style="bold cyan")
    table.add_column("Команда", style="green")
    table.add_column("Описание", style="white")
    table.add_row("/help", "Показать справку")
    table.add_row("/status", "Показать текущие настройки и состояние памяти")
    table.add_row("/clear", "Очистить историю диалога, сохранить entity-memory")
    table.add_row("/clear --all", "Очистить историю и entity-memory")
    table.add_row("/character <name>", f"Сменить характер: {', '.join(sorted(CHARACTER_PROMPTS.keys()))}")
    table.add_row("/memory <buffer|summary>", "Переключить стратегию памяти")
    table.add_row("/quit", "Выход")
    console.print(table)


def render_status(console: Console, assistant: SmartAssistant) -> None:
    snapshot: Dict[str, Union[str, int]] = assistant.status_snapshot()

    table = Table(title="Состояние ассистента", show_header=False, box=None)
    table.add_column("Параметр", style="bold cyan")
    table.add_column("Значение", style="white")

    for key, value in snapshot.items():
        table.add_row(key, str(value))

    console.print(table)

    entities: Dict[str, str] = assistant.memory.entities
    if entities:
        facts_table = Table(title="Entity Memory", show_header=True, header_style="bold magenta")
        facts_table.add_column("Ключ", style="magenta")
        facts_table.add_column("Значение", style="white")
        for key, value in sorted(entities.items()):
            facts_table.add_row(key, value)
        console.print(facts_table)


def render_response_meta(console: Console, response: AssistantResponse) -> None:
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("confidence", f"{response.confidence:.2f}")
    table.add_row("tokens", f"~{response.tokens_used}")
    console.print(table)


def handle_command(console: Console, assistant: SmartAssistant, raw_command: str) -> bool:
    command: str = raw_command.strip()

    if command == "/help":
        render_help(console)
        return True

    if command == "/status":
        render_status(console, assistant)
        return True

    if command == "/clear":
        assistant.clear_history()
        console.print("[bold green]✓ История диалога очищена[/bold green]")
        return True

    if command == "/clear --all":
        assistant.memory.clear_all()
        console.print("[bold green]✓ История и entity-memory очищены[/bold green]")
        return True

    if command == "/quit":
        console.print("[bold blue]До связи.[/bold blue]")
        raise SystemExit(0)

    if command.startswith("/character "):
        _, _, value = command.partition(" ")
        if value not in CHARACTER_PROMPTS:
            console.print(f"[bold red]Неизвестный характер:[/bold red] {value}")
            return True
        assistant.set_character(value)
        console.print(f"[bold green]✓ Характер изменён на:[/bold green] {value}")
        return True

    if command.startswith("/memory "):
        _, _, value = command.partition(" ")
        if value not in {"buffer", "summary"}:
            console.print(f"[bold red]Неизвестная стратегия памяти:[/bold red] {value}")
            return True
        assistant.set_memory_strategy(value)  # type: ignore[arg-type]
        console.print(f"[bold green]✓ Стратегия памяти изменена на:[/bold green] {value}")
        return True

    console.print("[bold red]Неизвестная команда.[/bold red] Используйте /help")
    return True


def main() -> None:
    args = build_arg_parser().parse_args()
    console = Console()

    try:
        assistant = SmartAssistant(
            model_name=args.model,
            fallback_model_name=args.fallback_model,
            temperature=args.temperature,
            character=args.character,
            memory_strategy=args.memory,
            console=console,
        )
    except Exception as exc:
        console.print(f"[bold red]Ошибка инициализации ассистента:[/bold red] {exc}")
        sys.exit(1)

    render_banner(console, assistant)
    render_help(console)

    while True:
        try:
            user_input: str = console.input("\n[bold white]> [/bold white]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold blue]До связи.[/bold blue]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(console, assistant, user_input)
            continue

        response: AssistantResponse = assistant.process(user_input)
        render_response_meta(console, response)


if __name__ == "__main__":
    main()
