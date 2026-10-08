import os
import base64
import mimetypes
import gradio as gr
from groq import Groq


# =========================================================
# GROQ
# =========================================================

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Wiens AI, a helpful multilingual AI assistant.

LANGUAGE:
- Automatically detect the user's language.
- Answer in the same language as the user.
- Russian, German and English are especially important.
- Other languages are also allowed when possible.

BEHAVIOR:
- Be helpful, clear and friendly.
- Explain difficult topics simply.
- Keep answers concise unless the user asks for detail.
- Remember the context of the current conversation.

INTERNET:
- You have access to browser search when available.
- Use web search when current or up-to-date information is useful.
- Examples: news, weather, prices, current events,
  companies, products, travel information and recent changes.
- Do not pretend that old model knowledge is current.
- When information comes from web research, make that clear.

IMAGES:
- When an image is provided, carefully inspect it.
- Read visible text using your vision capabilities.
- Explain documents simply.
- Translate visible text when requested.
- Answer questions about objects and information visible in images.
- Never invent details that cannot be seen.

You are called Wiens AI.
"""


# =========================================================
# IMAGE -> BASE64
# =========================================================

def image_to_data_url(file_path):

    mime_type, _ = mimetypes.guess_type(file_path)

    if not mime_type:
        mime_type = "image/jpeg"

    with open(file_path, "rb") as image_file:

        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return (
        f"data:{mime_type};base64,{encoded}"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

def add_history(messages, history):

    if not history:
        return

    for item in history:

        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if (
                role in ["user", "assistant"]
                and isinstance(content, str)
            ):

                messages.append({
                    "role": role,
                    "content": content
                })


        elif (
            isinstance(item, (list, tuple))
            and len(item) == 2
        ):

            user_part = item[0]
            assistant_part = item[1]

            if isinstance(user_part, str):

                messages.append({
                    "role": "user",
                    "content": user_part
                })

            if isinstance(assistant_part, str):

                messages.append({
                    "role": "assistant",
                    "content": assistant_part
                })


# =========================================================
# FILE PATH
# =========================================================

def get_file_path(file_item):

    if isinstance(file_item, str):
        return file_item

    if isinstance(file_item, dict):

        return (
            file_item.get("path")
            or file_item.get("name")
        )

    path = getattr(
        file_item,
        "path",
        None
    )

    if path:
        return path

    return getattr(
        file_item,
        "name",
        None
    )


# =========================================================
# IMAGE CHAT
# =========================================================

def vision_chat(user_text, files, history):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    add_history(
        messages,
        history
    )

    content = []

    if user_text.strip():

        content.append({
            "type": "text",
            "text": user_text
        })

    else:

        content.append({
            "type": "text",
            "text":
                "Analyze this image carefully and "
                "explain what you see."
        })


    # Maximum 3 images
    for file_item in files[:3]:

        file_path = get_file_path(
            file_item
        )

        if not file_path:
            continue

        data_url = image_to_data_url(
            file_path
        )

        content.append({
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        })


    messages.append({
        "role": "user",
        "content": content
    })


    response = client.chat.completions.create(

        model="qwen/qwen3.8-27b",

        messages=messages,

        temperature=0.7,

        max_completion_tokens=2000,

        reasoning_effort="none"
    )


    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================================================
# TEXT + INTERNET CHAT
# =========================================================

def internet_chat(user_text, history):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    add_history(
        messages,
        history
    )

    messages.append({
        "role": "user",
        "content": user_text
    })


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=messages,

        tools=[
            {
                "type": "browser_search"
            }
        ],

        reasoning_effort="medium",

        max_completion_tokens=2000
    )


    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================================================
# MAIN CHAT FUNCTION
# =========================================================

def chat(message, history):

    try:

        # MultimodalTextbox returns a dictionary
        if isinstance(message, dict):

            user_text = (
                message.get("text")
                or ""
            )

            files = (
                message.get("files")
                or []
            )

        else:

            user_text = str(message)

            files = []


        # IMAGE MODE
        if files:

            return vision_chat(
                user_text,
                files,
                history
            )


        # TEXT + INTERNET MODE
        if user_text.strip():

            return internet_chat(
                user_text,
                history
            )


        return "Напишите сообщение или прикрепите изображение."


    except Exception as e:

        return (
            "Wiens AI error: "
            + str(e)
        )


# =========================================================
# DESIGN
# =========================================================

css = """
body,
.gradio-container {

    background:
        radial-gradient(
            circle at top,
            #24164d 0%,
            #10101b 38%,
            #07080d 100%
        ) !important;

    color: white !important;
}


.gradio-container {

    max-width: 850px !important;

    margin: 0 auto !important;

    min-height: 100vh !important;
}


/* LOGO */

.logo-wrap {

    text-align: center;

    padding-top: 22px;
}


.logo-wrap img {

    width: 145px;

    height: 145px;

    object-fit: cover;

    border-radius: 32px;

    box-shadow:
        0 0 30px rgba(116,80,255,.35),
        0 15px 55px rgba(0,0,0,.45);
}


/* TITLE */

.app-title {

    text-align: center;

    font-size: 40px;

    font-weight: 800;

    margin-top: 14px;

    letter-spacing: -1px;
}


/* SUBTITLE */

.subtitle {

    text-align: center;

    color: #a6a6b5;

    font-size: 15px;

    margin-top: 5px;
}


/* FEATURES */

.features {

    text-align: center;

    color: #88889a;

    font-size: 12px;

    margin-top: 8px;
}


/* ONLINE */

.status-wrap {

    text-align: center;

    margin-top: 12px;

    margin-bottom: 24px;
}


.status {

    display: inline-block;

    padding: 6px 13px;

    border-radius: 20px;

    background:
        rgba(72,255,150,.07);

    border:
        1px solid rgba(72,255,150,.18);

    color: #77ffa9;

    font-size: 12px;
}


/* CHAT */

.chatbot {

    background:
        rgba(17,17,25,.94) !important;

    border:
        1px solid #2c2c39 !important;

    border-radius:
        24px !important;

    overflow: hidden !important;

    box-shadow:
        0 20px 60px rgba(0,0,0,.35);
}


/* INPUT */

textarea {

    border-radius:
        18px !important;
}


/* BUTTON */

button {

    border-radius:
        16px !important;

    font-weight:
        600 !important;
}


/* FOOTER */

.footer {

    text-align: center;

    color: #686876;

    font-size: 12px;

    padding:
        17px 0 25px 0;
}
"""


# =========================================================
# APP
# =========================================================

with gr.Blocks(

    css=css,

    title="Wiens AI",

    theme=gr.themes.Base()

) as demo:


    gr.HTML("""
    <div class="logo-wrap">

        <img src="/gradio_api/file=logo.png.PNG">

    </div>


    <div class="app-title">

        Wiens AI

    </div>


    <div class="subtitle">

        Русский · Deutsch · English

    </div>


    <div class="features">

        🌐 Internet · 📷 Vision · 🧠 AI

    </div>


    <div class="status-wrap">

        <span class="status">

            ● AI Online

        </span>

    </div>
    """)


    chatbot = gr.Chatbot(

        height=500,

        elem_classes=["chatbot"],

        placeholder="""

        <div style="
            text-align:center;
            opacity:.70;
        ">

            <div style="
                font-size:32px;
            ">

                ✦

            </div>


            <div style="
                font-size:20px;
                font-weight:600;
                margin-top:10px;
            ">

                Чем я могу помочь?

            </div>


            <div style="
                font-size:13px;
                margin-top:7px;
                opacity:.7;
            ">

                Спросите что-нибудь
                или прикрепите фотографию

            </div>

        </div>

        """
    )


    multimodal_input = gr.MultimodalTextbox(

        placeholder=
            "Напишите сообщение или добавьте фото...",

        file_types=[
            "image"
        ],

        file_count="multiple",

        sources=[
            "upload"
        ],

        submit_btn="➤",

        show_label=False
    )


    gr.ChatInterface(

        fn=chat,

        chatbot=chatbot,

        textbox=multimodal_input,

        multimodal=True
    )


    gr.HTML("""

    <div class="footer">

        Wiens AI
        · Internet
        · Vision
        · Powered by Groq

    </div>

    """)


# =========================================================
# START SERVER
# =========================================================

demo.launch(

    server_name="0.0.0.0",

    server_port=int(
        os.environ.get(
            "PORT",
            10000
        )
    ),

    allowed_paths=[
        "logo.png.PNG"
    ]
)
