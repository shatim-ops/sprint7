"""Клиент к языковой модели.

Работаем через OpenAI-совместимый протокол: так один и тот же код ходит
и в Cloud.ru Foundation Models, и в OpenAI, и в локальный vLLM.
Меняется только LLM_BASE_URL в .env.
"""

from dataclasses import dataclass


@dataclass
class LlmReply:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatModel:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float, max_tokens: int):
        from openai import OpenAI

        if not api_key:
            raise RuntimeError(
                "Не задан LLM_API_KEY. Скопируйте .env.example в .env и впишите ключ."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list) -> LlmReply:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            # Рассуждающие модели иногда кладут весь вывод в отдельное поле,
            # а при обрыве по лимиту возвращают пустой content. Разбираем оба случая.
            text = (getattr(choice.message, "reasoning_content", "") or "").strip()
        if not text:
            raise RuntimeError(
                f"Модель {self.model} вернула пустой ответ, "
                f"finish_reason={getattr(choice, 'finish_reason', 'неизвестен')}. "
                "Чаще всего это упёршийся LLM_MAX_TOKENS."
            )
        usage = getattr(response, "usage", None)
        return LlmReply(
            text=text,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class OfflineModel:
    """Заглушка для тестов и сборки без сети.

    Никакой генерации здесь нет: ответ собирается из найденных фрагментов.
    Нужна только чтобы прогонять пайплайн там, где нет ключа и выхода в интернет.
    """

    model = "offline-extractive"

    def complete(self, messages: list) -> LlmReply:
        context = messages[-1]["content"]
        body = context.split("КОНТЕКСТ:", 1)[-1].split("Вопрос:")[0].strip()
        question = context.rsplit("Вопрос:", 1)[-1].strip()
        if not body:
            text = (
                "Рассуждение:\n1. Контекст пуст, опираться не на что.\n"
                "Ответ: Я не знаю.\nИсточники: []"
            )
        else:
            first = body.split("\n\n")[0]
            snippet = first.split("\n", 1)[-1].strip()
            title = first.split("документ: ", 1)[-1].split(" (")[0]
            text = (
                "Рассуждение:\n"
                f"1. По запросу \"{question}\" найден фрагмент документа {title}.\n"
                "2. Ответ формирую по этому фрагменту, других данных в контексте нет.\n"
                f"Ответ: {snippet[:400]}\n"
                f"Источники: [{title}]"
            )
        return LlmReply(text=text, model=self.model)


def build_model(settings):
    if settings.llm_api_key:
        return ChatModel(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    return OfflineModel()
