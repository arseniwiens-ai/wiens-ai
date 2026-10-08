import os
import gradio as gr
from groq import Groq


# -----------------------------
# GROQ CONNECTION
# -----------------------------

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# -----------------------------
# WIENS AI SYSTEM PROMPT
# -----------------------------

SYSTEM_PROMPT = """
You are Wiens AI, a helpful multilingual AI assistant.

Automatically detect the language of the user.

If the user writes in Russian, answer in Russian.
If the user writes in German, answer in German.
If the user writes in English, answer in English.
If the user writes in another language, answer in that language when possible.

Be helpful, clear, friendly, and concise.
Explain difficult things in simple language.
Remember the context of the current conversation.
"""


# -----------------------------
# CHAT FUNCTION
# -----------------------------

def chat(message, history):
    try:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")

                if role in ["user", "assistant"] and isinstance(content, str):
                    messages.append({
                        "role": role,
                        "content": content
                    })

            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if item[0]:
                    messages.append({
                        "role": "user",
                        "content": str(item[0])
                    })

                if item[1]:
                    messages.append({
                        "role": "assistant",
                        "content": str(item[1])
                    })

        messages.append({
            "role": "user",
            "content": message
        })

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI connection error: {str(e)}"


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
            #10101b 35%,
            #07080d 75%
        ) !important;

    color: #ffffff !important;
}

.gradio-container {
    max-width: 850px !important;
    margin: 0 auto !important;
    min-height: 100vh !important;
}


/* LOGO */

.logo-wrap {
    text-align: center;
    padding-top: 24px;
}

.logo-wrap img {
    width: 145px;
    height: 145px;

    object-fit: cover;

    border-radius: 32px;

    box-shadow:
        0 0 30px rgba(116, 80, 255, 0.35),
        0 15px 55px rgba(0, 0, 0, 0.45);
}


/* TITLE */

.app-title {
    text-align: center;

    font-size: 40px;
    font-weight: 800;

    margin-top: 14px;

    letter-spacing: -1px;
}


/* LANGUAGES */

.subtitle {
    text-align: center;

    color: #a6a6b5;

    font-size: 15px;

    margin-top: 5px;
    margin-bottom: 12px;
}


/* ONLINE STATUS */

.status-wrap {
    text-align: center;
    margin-bottom: 24px;
}

.status {
    display: inline-block;

    padding: 6px 13px;

    border-radius: 20px;

    background: rgba(72, 255, 150, 0.07);

    border:
        1px solid rgba(72, 255, 150, 0.18);

    color: #77ffa9;

    font-size: 12px;
}


/* CHAT WINDOW */

.chatbot {
    background:
        rgba(17, 17, 25, 0.94) !important;

    border:
        1px solid #2c2c39 !important;

    border-radius:
        24px !important;

    overflow:
        hidden !important;

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.35);
}


/* TEXT INPUT */

textarea {
    border-radius:
        18px !important;
}


/* BUTTONS */

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
                Задайте вопрос Wiens AI
            </div>

        </div>
        """
    )

    gr.ChatInterface(
        fn=chat,

        chatbot=chatbot,

        textbox=gr.Textbox(
            placeholder="Напишите сообщение...",
            container=False
        ),

        submit_btn="➤"
    )

    gr.HTML("""
    <div class="footer">
        Wiens AI · Powered by Groq
    </div>
    """)


# -----------------------------
# START SERVER
# -----------------------------

demo.launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get("PORT", 10000)
    ),
    allowed_paths=["logo.png.PNG"]
)
