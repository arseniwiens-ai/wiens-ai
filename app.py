import os
import base64
import mimetypes
import gradio as gr
from groq import Groq


# -----------------------------
# GROQ CONNECTION
# -----------------------------

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# -----------------------------
# WIENS AI
# -----------------------------

SYSTEM_PROMPT = """
You are Wiens AI, a helpful multilingual AI assistant.

Automatically detect the language of the user.

If the user writes in Russian, answer in Russian.
If the user writes in German, answer in German.
If the user writes in English, answer in English.
If the user writes in another language, answer in that language when possible.

You can also analyze images.

When the user sends an image:
- Carefully inspect the image.
- Read visible text when possible.
- Explain documents in simple language.
- Translate text if the user asks.
- Answer questions about the image.
- Do not invent details that are not visible.

Be helpful, clear, friendly, and concise.
Explain difficult things simply.
"""


# -----------------------------
# IMAGE TO BASE64
# -----------------------------

def image_to_data_url(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if not mime_type:
        mime_type = "image/jpeg"

    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# -----------------------------
# HISTORY
# -----------------------------

def add_history(messages, history):

    for item in history:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ["user", "assistant"]:
            continue

        # Keep only text from previous messages.
        # This avoids repeatedly sending old images.
        if isinstance(content, str):
            messages.append({
                "role": role,
                "content": content
            })

        elif isinstance(content, list):

            text_parts = []

            for part in content:

                if isinstance(part, str):
                    text_parts.append(part)

                elif isinstance(part, dict):

                    if part.get("type") == "text":
                        text = part.get("text")

                        if text:
                            text_parts.append(text)

            if text_parts:
                messages.append({
                    "role": role,
                    "content": "\n".join(text_parts)
                })


# -----------------------------
# CHAT
# -----------------------------

def chat(message, history):

    try:

        # Multimodal Gradio message
        if isinstance(message, dict):

            user_text = message.get("text") or ""
            files = message.get("files") or []

        else:

            user_text = str(message)
            files = []


        # -------------------------
        # IMAGE MESSAGE
        # -------------------------

        if files:

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            add_history(messages, history)

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
                        "Please analyze this image and explain what you see."
                })


            # Groq Qwen supports up to 3 images
            for file_item in files[:3]:

                if isinstance(file_item, str):
                    file_path = file_item

                elif isinstance(file_item, dict):
                    file_path = (
                        file_item.get("path")
                        or file_item.get("name")
                    )

                else:
                    file_path = getattr(
                        file_item,
                        "path",
                        None
                    )

                    if not file_path:
                        file_path = getattr(
                            file_item,
                            "name",
                            None
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
                max_completion_tokens=1500
            )


        # -------------------------
        # TEXT MESSAGE
        # -------------------------

        else:

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            add_history(messages, history)

            messages.append({
                "role": "user",
                "content": user_text
            })


            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )


        return response.choices[0].message.content


    except Exception as e:

        return (
            "Wiens AI error: "
            + str(e)
        )


# -----------------------------
# DESIGN
# -----------------------------

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


/* STATUS */

.status-wrap {

    text-align: center;

    margin-top: 10px;

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


# -----------------------------
# APP
# -----------------------------

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
                margin-top:6px;
                opacity:.7;
            ">

                Напишите сообщение
                или прикрепите изображение

            </div>

        </div>

        """
    )


    multimodal_input = gr.MultimodalTextbox(

        placeholder="Сообщение или фото...",

        file_types=["image"],

        file_count="multiple",

        sources=["upload"],

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

        Wiens AI · Vision enabled · Powered by Groq

    </div>

    """)


# -----------------------------
# START
# -----------------------------

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
