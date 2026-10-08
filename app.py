import os
import gradio as gr
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are Wiens AI, a helpful multilingual AI assistant.

Automatically detect the language of the user.

If the user writes in Russian, answer in Russian.
If the user writes in German, answer in German.
If the user writes in English, answer in English.
If the user writes in another language, answer in that language when possible.

Be helpful, clear, friendly, and concise.
Explain difficult things in simple language.
"""

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


css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

h1 {
    text-align: center;
    font-size: 42px !important;
    margin-bottom: 5px !important;
}

.subtitle {
    text-align: center;
    opacity: 0.7;
    font-size: 20px;
    margin-bottom: 25px;
}
"""


with gr.Blocks(
    css=css,
    title="Wiens AI"
) as demo:

    gr.Markdown("# ✦ Wiens AI")

    gr.HTML(
        '<div class="subtitle">'
        'Русский · Deutsch · English'
        '</div>'
    )

    gr.ChatInterface(
        fn=chat,
        examples=[
            "Привет! Что ты умеешь?",
            "Hallo! Wer bist du?",
            "Hello! What can you do?"
        ]
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
