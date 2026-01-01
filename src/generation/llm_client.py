"""LLM client for answer generation."""

from typing import Optional
from openai import OpenAI
from loguru import logger

from src.config import settings


class LLMClient:
    """OpenAI LLM client for answer generation."""

    SYSTEM_PROMPT = """You are a research assistant specialized in AI/ML papers.
Answer questions based ONLY on the provided context from research papers.

CRITICAL INSTRUCTIONS:
1. If the context doesn't contain sufficient information to answer, say so explicitly.
2. Always cite your sources using the format [Source N] where N matches the source number in the context.
3. Use direct quotes sparingly, paraphrasing when possible.
4. Structure complex answers with clear organization.
5. Distinguish between what the paper explicitly states vs. your interpretation.
6. Be precise and technical when discussing paper content."""

    def __init__(self, api_key: str = None, model: str = None):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.llm_model
        logger.info(f"Initialized LLM client with model: {self.model}")

    def generate(
        self,
        question: str,
        context: str,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> str:
        """
        Generate an answer based on question and context.

        Args:
            question: The user's question
            context: Retrieved context from documents
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer string
        """
        user_prompt = f"""Context from research papers:

{context}

---

Question: {question}

Please provide a comprehensive answer based on the context above. Include citations to the sources using [Source N] format."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt or self.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.max_output_tokens
            )

            answer = response.choices[0].message.content
            logger.debug(f"Generated answer: {len(answer)} chars")
            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_with_messages(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response with custom message list."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.max_output_tokens
        )
        return response.choices[0].message.content
