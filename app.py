import os
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=os.environ.get("HF_TOKEN")
)

MODEL = "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = """
You are Wiens AI, a helpful multilingual AI assistant.

Always answer in the same language as the user.
You understand Russian, German and English.
Give clear, useful and natural answers.
Remember the context of the current conversation.
If you do not know something, say so instead of inventing facts.
"""

def respond(message, history):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    for item in history:
        if isinstance(item, dict):
            messages.append(item)

    messages.append({
        "role": "user",
        "content": message
    })

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=700,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"AI connection error: {error}"


css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

#title {
    text-align: center;
    margin-bottom: 0;
}

#subtitle {
    text-align: center;
    opacity: 0.7;
    margin-bottom: 20px;
}
"""


with gr.Blocks(
    title="Wiens AI",
    css=css,
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        "# ✦ Wiens AI",
        elem_id="title"
    )

    gr.Markdown(
        "Русский • Deutsch • English",
        elem_id="subtitle"
    )

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            height=520,
            placeholder="<h3>Wiens AI</h3><p>Чем могу помочь?</p>"
        ),
        textbox=gr.Textbox(
            placeholder="Напишите сообщение...",
            container=False
        ),
        submit_btn="➤"
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
