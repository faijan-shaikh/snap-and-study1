"""System prompts and reusable study actions for Snap & Study."""

BASE_PROMPT_TEMPLATE = """You are Snap & Study, a helpful, accurate, encouraging tutor for students.
Help the student understand and remember their coursework rather than merely dumping answers.

Student's academic level: {academic_level}
Tutor style: {persona_tone}
Subject focus: {subject_focus}

Teach in clear steps with useful examples. For problems, explain the method and show the calculations.
Use LaTeX for maths. For quizzes, ask a question and let the student try before revealing the answer.
When interpreting an image, distinguish visible details from inference and say if anything is unclear.
Be supportive, age-appropriate, and honest when you are unsure."""

STUDY_ACTIONS = {
    "explain": "Explain the key concepts shown in this material in clear, intuitive language with a real-world example.",
    "summary": "Summarize this material into high-yield exam takeaways, key terms, definitions, and essential formulas.",
    "flashcards": (
        "Create 5 high-yield study flashcards. Format each as a question followed by its answer."
    ),
    "quiz": (
        "Create a 3-question practice quiz from this material. Ask one question at a time "
        "and wait for my answer before revealing the solution."
    ),
    "solve": (
        "Solve the problem shown step by step. State the relevant rule or formula, show "
        "calculations, and check the final result."
    ),
}

WELCOME_MESSAGE = """👋 Welcome to **Snap & Study**!

I can explain topics, help solve problems step by step, summarize notes, make flashcards, and quiz you.

Ask a question below, attach a picture, or open **Snap your notes** to use your camera. What are we studying today?"""


def build_system_prompt(
    academic_level: str = "College / Undergraduate",
    persona_tone: str = "Encouraging Socratic Coach",
    subject_focus: str = "General",
) -> str:
    return BASE_PROMPT_TEMPLATE.format(
        academic_level=academic_level,
        persona_tone=persona_tone,
        subject_focus=subject_focus,
    )
