# AgentSystemsHSE2026-SmartAssistant
Agent systems home work project materials:
# Smart Assistant (LangChain CLI)

CLI-ассистент "с характером" на базе LangChain + Pydantic + Rich.

## Возможности
- Классификация запросов (LCEL + PydanticOutputParser)
- Роутинг через RunnableBranch
- Поддержка характеров (friendly, professional, sarcastic, pirate)
- Память: buffer и summary
- Entity memory (имя, предпочтения)
- Streaming-ответы в терминале
- CLI-команды (/character, /memory, /clear, /status)

## Запуск

```bash
python smart_assistant.py --model llama3 --fallback-model llama3 # локально
