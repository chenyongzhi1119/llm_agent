"""Personalized study tutor tool - knowledge explanation and quiz generation"""

from src.tools.base import Tool
from config.settings import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from openai import OpenAI


class StudyTutorTool(Tool):
    name = "study_tutor"
    description = "Study tutor: explain topics, generate quizzes, or grade answers."
    parameters = {
        "action": {"description": "explain | quiz | grade", "required": True},
        "topic": {"description": "subject to study", "required": True},
        "level": {"description": "beginner|intermediate|advanced, default intermediate", "required": False},
        "student_answer": {"description": "answer to grade (for grade action only)", "required": False},
    }

    def execute(self, **kwargs) -> str:
        action = kwargs.get("action", "")
        topic = kwargs.get("topic", "")
        level = kwargs.get("level", "intermediate")
        student_answer = kwargs.get("student_answer", "")

        if not action or not topic:
            return "Error: 'action' and 'topic' are required"

        if action == "explain":
            return self._explain(topic, level)
        elif action == "quiz":
            return self._quiz(topic, level)
        elif action == "grade":
            if not student_answer:
                return "Error: 'student_answer' is required for grading"
            return self._grade(topic, student_answer, level)
        else:
            return f"Error: unknown action '{action}'. Use: explain, quiz, or grade"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        try:
            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            return response.choices[0].message.content or "No response"
        except Exception as e:
            return f"LLM error: {e}"

    def _explain(self, topic: str, level: str) -> str:
        system = (
            f"You are a patient tutor. Explain concepts at the {level} level. "
            f"Use clear language, examples, and analogies. Structure your explanation with sections."
        )
        return self._call_llm(system, f"Explain this topic: {topic}")

    def _quiz(self, topic: str, level: str) -> str:
        system = (
            f"You are a quiz generator. Create 3 questions at the {level} level. "
            f"Mix question types: multiple choice, short answer, and true/false. "
            f"Include the correct answers at the end."
        )
        return self._call_llm(system, f"Generate quiz questions about: {topic}")

    def _grade(self, topic: str, student_answer: str, level: str) -> str:
        system = (
            f"You are a grading assistant. Evaluate the student's answer at the {level} level. "
            f"Provide: 1) Score (0-100), 2) What's correct, 3) What's wrong or missing, 4) Suggestions to improve."
        )
        user = f"Topic: {topic}\n\nStudent's answer:\n{student_answer}"
        return self._call_llm(system, user)
