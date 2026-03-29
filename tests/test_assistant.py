from langchain_core.messages import AIMessage, HumanMessage

from smart_assistant import MemoryManager, RequestType, Classification


def test_classification_model_accepts_valid_payload() -> None:
    payload = Classification(
        request_type=RequestType.QUESTION,
        confidence=0.91,
        reasoning="Похоже на вопрос",
    )
    assert payload.request_type == RequestType.QUESTION
    assert payload.confidence == 0.91


def test_buffer_memory_trims_old_messages() -> None:
    memory = MemoryManager(strategy="buffer", max_messages=4)
    for i in range(4):
        memory.add_turn(f"user-{i}", f"ai-{i}")

    history = memory.get_prompt_history()
    assert len(history) == 4
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[-1], AIMessage)
    assert "user-2" in history[0].content
    assert "ai-3" in history[-1].content


def test_entity_memory_extracts_name_and_language() -> None:
    memory = MemoryManager(strategy="buffer")
    memory.add_user_message("Привет, меня зовут Даша")
    memory.add_user_message("Мой любимый язык — Python")

    entities = memory.entities
    assert entities["name"].lower() == "даша"
    assert entities["favorite_language"].lower() == "python"


def test_clear_history_keeps_entities() -> None:
    memory = MemoryManager(strategy="buffer")
    memory.add_turn("Меня зовут Алексей", "Привет, Алексей")
    memory.clear_history()

    assert memory.message_count == 0
    assert memory.entities["name"].lower() == "алексей"
