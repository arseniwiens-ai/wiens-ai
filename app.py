import gradio as gr

def respond(message, history):
    message_lower = message.lower()

    if any(word in message_lower for word in ["привет", "hello", "hallo", "hi"]):
        return "Привет! 👋 Я Wiens AI. Чем могу помочь?"

    if "кто ты" in message_lower or "who are you" in message_lower:
        return "Я Wiens AI — небольшой AI-ассистент."

    if "как дела" in message_lower:
        return "У меня всё отлично 😄 А у тебя?"

    return f"Ты написал: {message}"

demo = gr.ChatInterface(
    fn=respond,
    title="🤖 Wiens AI",
    description="Русский • Deutsch • English"
)

demo.launch(
    server_name="0.0.0.0",
    server_port=10000
)
