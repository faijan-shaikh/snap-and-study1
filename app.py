import base64
import io
import json
import os
import re
import time
from typing import Any, Generator

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from groq import APIError as GroqAPIError
from groq import Groq
from PIL import Image, UnidentifiedImageError
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client as TwilioClient

from prompts import STUDY_ACTIONS, WELCOME_MESSAGE, build_system_prompt

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"
MAX_RECENT_IMAGES = 2

st.set_page_config(
    page_title="Snap & Study",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1050px; padding-top: 2rem; }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


@st.cache_resource
def get_twilio_client(account_sid: str, auth_token: str) -> TwilioClient:
    return TwilioClient(account_sid, auth_token)


def get_setting(name: str) -> str:
    try:
        secret_value = st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        secret_value = ""
    return str(secret_value or os.environ.get(name, "")).strip()


def image_data_url(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("That image could not be opened. Try a PNG, JPG, or WEBP image.") from error

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_chat_payload(
    history: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[list[dict[str, Any]], bool]:
    image_indexes = [
        index for index, message in enumerate(history) if message.get("kind") == "image"
    ]
    recent_image_indexes = set(image_indexes[-MAX_RECENT_IMAGES:])
    payload: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    has_image = False
    index = 0

    while index < len(history):
        message = history[index]
        role = message["role"]

        if role == "assistant":
            payload.append({"role": "assistant", "content": str(message["content"])})
            index += 1
            continue

        parts: list[dict[str, Any]] = []
        while index < len(history) and history[index]["role"] == "user":
            current = history[index]
            if current.get("kind") == "image":
                has_image = True
                if index in recent_image_indexes:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(current["content"])},
                        }
                    )
                else:
                    parts.append({"type": "text", "text": "[Earlier study image]"})
            elif current.get("content"):
                parts.append({"type": "text", "text": str(current["content"])})
            index += 1

        if len(parts) == 1 and parts[0]["type"] == "text":
            payload.append({"role": "user", "content": parts[0]["text"]})
        elif parts:
            payload.append({"role": "user", "content": parts})

    return payload, has_image


def stream_answer(
    client: Groq,
    payload: list[dict[str, Any]],
    model: str,
) -> Generator[str, None, None]:
    response = client.chat.completions.create(
        model=model,
        messages=payload,
        temperature=0.4,
        stream=True,
    )
    for chunk in response:
        if chunk.choices:
            text = chunk.choices[0].delta.content
            if text:
                yield text


def normalize_whatsapp_address(number: str) -> str:
    number = number.strip()
    if number.lower().startswith("whatsapp:"):
        number = number.split(":", 1)[1]
    number = re.sub(r"[\s()-]", "", number)
    if not re.fullmatch(r"\+[1-9]\d{7,14}", number):
        raise ValueError("Enter a WhatsApp number in international format, such as +14155552671.")
    return f"whatsapp:{number}"


def create_study_recap(
    client: Groq,
    history: list[dict[str, Any]],
    system_prompt: str,
    student_name: str,
) -> str:
    transcript = []
    for message in history:
        if message.get("kind") == "text":
            role = "Student" if message["role"] == "user" else "Tutor"
            transcript.append(f"{role}: {message['content']}")
    if not transcript:
        raise ValueError("Ask a study question before creating a recap.")

    result = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Create a concise study recap for {student_name}. Include the key ideas, "
                    "important formulas or definitions, and useful next steps. Do not invent facts. "
                    "Keep it plain text and under 1,400 characters for WhatsApp.\n\n"
                    f"Chat transcript:\n{chr(10).join(transcript)[-10000:]}"
                ),
            },
        ],
        temperature=0.2,
    )
    recap = result.choices[0].message.content
    if not recap:
        raise RuntimeError("Groq returned an empty recap.")
    return recap.strip()[:1500]


def send_whatsapp_recap(
    account_sid: str,
    auth_token: str,
    sender: str,
    recipient: str,
    content_sid: str,
    student_name: str,
    recap: str,
) -> str:
    client = get_twilio_client(account_sid, auth_token)
    message = client.messages.create(
        from_=normalize_whatsapp_address(sender),
        to=normalize_whatsapp_address(recipient),
        content_sid=content_sid,
        content_variables=json.dumps({"1": student_name, "2": recap}, ensure_ascii=False),
    )
    return message.sid


if "messages" not in st.session_state:
    st.session_state.messages = []
if "snap_count" not in st.session_state:
    st.session_state.snap_count = 0
if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = ""
if "camera_bytes" not in st.session_state:
    st.session_state.camera_bytes = None

api_key_from_config = get_setting("GROQ_API_KEY")
account_sid = get_setting("TWILIO_ACCOUNT_SID")
auth_token = get_setting("TWILIO_AUTH_TOKEN")
whatsapp_sender = get_setting("TWILIO_WHATSAPP_FROM")
content_sid = get_setting("TWILIO_CONTENT_SID")

with st.sidebar:
    st.header("⚙️ Study settings")
    st.text_input(
        "Groq API key",
        type="password",
        key="api_key_input",
        placeholder="Loaded from secrets" if api_key_from_config else "Paste your Groq API key",
        help="You can also set GROQ_API_KEY in .streamlit/secrets.toml or as an environment variable.",
    )
    api_key = st.session_state.api_key_input.strip() or api_key_from_config
    academic_level = st.selectbox(
        "Academic level",
        [
            "Middle School",
            "High School (AP / IB / CBSE / GCSE)",
            "College / Undergraduate",
            "Graduate / Professional",
            "Competitive Exams (SAT / JEE / NEET / GRE)",
        ],
        index=1,
    )
    persona = st.selectbox(
        "Tutor style",
        [
            "Encouraging Socratic Coach",
            "Peer Study Buddy",
            "Detailed Professor",
            "High-Yield Crammer",
        ],
    )
    subject = st.selectbox(
        "Subject",
        [
            "General / Multi-disciplinary",
            "Mathematics & Statistics",
            "Physics & Engineering",
            "Chemistry & Biochemistry",
            "Biology",
            "Computer Science & Coding",
            "Economics & Business",
            "History & Social Sciences",
        ],
    )

    st.divider()
    st.metric("Snaps analyzed", st.session_state.snap_count)

    if st.session_state.messages:
        transcript = ["# Snap & Study Notes\n"]
        for message in st.session_state.messages:
            if message["kind"] == "text":
                role = "Student" if message["role"] == "user" else "Snap & Study"
                transcript.append(f"## {role}\n{message['content']}\n")
            elif message["kind"] == "image":
                transcript.append("## Student\n[Study image attached]\n")
        st.download_button(
            "📥 Export notes",
            data="\n".join(transcript),
            file_name=f"snap_study_notes_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.subheader("📲 WhatsApp recap")
    student_name = st.text_input("Student name", key="student_name")
    recipient = st.text_input(
        "WhatsApp number",
        key="whatsapp_number",
        placeholder="+14155552671",
    )
    whatsapp_ready = all((account_sid, auth_token, whatsapp_sender, content_sid))
    send_recap = st.button(
        "Send study recap",
        disabled=not (whatsapp_ready and bool(st.session_state.messages)),
        use_container_width=True,
    )
    if not whatsapp_ready:
        st.caption("Add Twilio credentials in Streamlit secrets to enable WhatsApp.")

    if st.button("🔄 Start fresh chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.snap_count = 0
        st.session_state.camera_bytes = None
        st.rerun()

st.title("📚 Snap & Study")
st.caption("Your AI study buddy. Ask questions, snap your notes, and learn at your pace.")

if not api_key:
    st.info("Add your Groq API key in the sidebar or `.streamlit/secrets.toml` to start chatting.")
    st.markdown("[Get a Groq API key](https://console.groq.com/keys)")
    st.stop()

groq_client = get_groq_client(api_key)
system_prompt = build_system_prompt(academic_level, persona, subject)

if send_recap:
    if not student_name.strip() or not recipient.strip():
        st.sidebar.error("Enter your name and WhatsApp number before sending.")
    else:
        try:
            with st.spinner("Creating and sending your recap..."):
                recap = create_study_recap(
                    groq_client,
                    st.session_state.messages,
                    system_prompt,
                    student_name.strip(),
                )
                send_whatsapp_recap(
                    account_sid,
                    auth_token,
                    whatsapp_sender,
                    recipient,
                    content_sid,
                    student_name.strip(),
                    recap,
                )
            st.sidebar.success("Your study recap was sent.")
        except (GroqAPIError, TwilioRestException, ValueError, RuntimeError) as error:
            st.sidebar.error(f"Couldn't send the recap: {error}")

with st.expander("📸 Snap your notes", expanded=False):
    camera_image = st.camera_input("Take a clear photo of notes, a textbook, or a problem")
    if camera_image is not None:
        st.session_state.camera_bytes = camera_image.getvalue()
        st.image(st.session_state.camera_bytes, caption="Study material")
        camera_action = st.selectbox(
            "What should I do with this image?",
            ["Explain", "Summarize", "Make flashcards", "Quiz me", "Solve the problem"],
        )
        if st.button("Analyze this snap", key="analyze_camera_snap"):
            prompt_key = {
                "Explain": "explain",
                "Summarize": "summary",
                "Make flashcards": "flashcards",
                "Quiz me": "quiz",
                "Solve the problem": "solve",
            }[camera_action]
            st.session_state.messages.extend(
                [
                    {
                        "role": "user",
                        "kind": "image",
                        "content": st.session_state.camera_bytes,
                        "mime": "image/jpeg",
                    },
                    {"role": "user", "kind": "text", "content": STUDY_ACTIONS[prompt_key]},
                ]
            )
            st.session_state.snap_count += 1
            st.session_state.camera_bytes = None
            st.rerun()

quick_actions = st.columns(4)
quick_prompts = [
    "Explain the last topic using a simple everyday example.",
    "Create 5 useful flashcards from our recent study discussion.",
    "Give me a 3-question quiz. Ask one question at a time and wait for my answer.",
    "List the important formulas and define each variable.",
]
for column, label, prompt in zip(
    quick_actions,
    ["💡 Explain", "📇 Flashcards", "🎯 Quiz me", "📐 Formulas"],
    quick_prompts,
):
    if column.button(label, use_container_width=True):
        st.session_state.pending_prompt = prompt

if st.session_state.messages:
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "📚"
        with st.chat_message(message["role"], avatar=avatar):
            if message["kind"] == "image":
                st.image(message["content"], caption="Study material")
            else:
                st.markdown(str(message["content"]))
else:
    with st.chat_message("assistant", avatar="📚"):
        st.markdown(WELCOME_MESSAGE)

chat_input = st.chat_input(
    "Ask a study question or attach a photo of your notes...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"],
)

user_text = st.session_state.pop("pending_prompt", "")
uploaded_image = None
if chat_input:
    user_text = chat_input.text.strip() if chat_input.text else ""
    if chat_input.files:
        uploaded_image = chat_input.files[0]

if user_text or uploaded_image is not None:
    if uploaded_image is not None:
        st.session_state.messages.append(
            {
                "role": "user",
                "kind": "image",
                "content": uploaded_image.getvalue(),
                "mime": uploaded_image.type or "image/jpeg",
            }
        )
        st.session_state.snap_count += 1
        if not user_text:
            user_text = "Explain the important study concepts, labels, and formulas in this image."
    if user_text:
        st.session_state.messages.append(
            {"role": "user", "kind": "text", "content": user_text}
        )
    payload, has_image = build_chat_payload(st.session_state.messages, system_prompt)
    model = VISION_MODEL if has_image else TEXT_MODEL

    with st.chat_message("assistant", avatar="📚"):
        try:
            answer = st.write_stream(stream_answer(groq_client, payload, model))
            st.session_state.messages.append(
                {"role": "assistant", "kind": "text", "content": answer}
            )
        except (GroqAPIError, ValueError, RuntimeError) as error:
            st.error(f"Couldn't get a study response: {error}")