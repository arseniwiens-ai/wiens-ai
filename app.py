import os
import tempfile
import base64

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from groq import Groq


app = Flask(__name__)


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


SYSTEM_PROMPT = """
You are Wiens AI, a helpful multilingual AI assistant.

Automatically detect the language of the user
and answer in the same language.

Russian, German and English are especially important.

Be helpful, clear, friendly and concise.
Explain difficult things simply.

Your name is Wiens AI.

You have access to browser search when using the
main text model.

Use browser search when the user asks about current,
recent, changing, or internet-based information.

Never claim that you searched the internet unless
browser search was actually used.
"""


VISION_PROMPT = """
You are Wiens AI with image understanding.

Analyze the image carefully.

Answer in the same language as the user's question.

Russian, German and English are especially important.

You can:
- describe images
- read text from images
- explain screenshots
- analyze documents
- answer questions about visible objects
- translate visible text
- help understand errors shown in screenshots

If something cannot be determined reliably from the
image, say so instead of guessing.
"""


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route("/service-worker.js")
def service_worker():
    response = send_from_directory(
        "static",
        "service-worker.js",
        mimetype="application/javascript"
    )

    response.headers["Cache-Control"] = "no-cache"

    return response


@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():
    try:
        data = request.get_json(
            silent=True
        ) or {}

        message = str(
            data.get(
                "message",
                ""
            )
        ).strip()

        history = data.get(
            "history",
            []
        )

        if not message:
            return jsonify({
                "error": "Message is empty"
            }), 400

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        if isinstance(
            history,
            list
        ):
            for item in history:
                if not isinstance(
                    item,
                    dict
                ):
                    continue

                role = item.get(
                    "role"
                )

                content = item.get(
                    "content"
                )

                if (
                    role in [
                        "user",
                        "assistant"
                    ]
                    and
                    isinstance(
                        content,
                        str
                    )
                ):
                    messages.append({
                        "role": role,
                        "content": content
                    })

        messages.append({
            "role": "user",
            "content": message
        })

        response = (
            client
            .chat
            .completions
            .create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                tools=[
                    {
                        "type": "browser_search"
                    }
                ]
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        print(
            "Wiens AI chat error:",
            e
        )

        return jsonify({
            "error":
                "Wiens AI could not answer: "
                + str(e)
        }), 500


@app.route(
    "/api/vision",
    methods=["POST"]
)
def vision():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "Image is missing"
            }), 400

        image = request.files[
            "image"
        ]

        question = str(
            request.form.get(
                "message",
                ""
            )
        ).strip()

        if not question:
            question = (
                "Describe this image "
                "and explain what you see."
            )

        image_bytes = image.read()

        if not image_bytes:
            return jsonify({
                "error": "Image is empty"
            }), 400

        if (
            len(image_bytes)
            >
            20 * 1024 * 1024
        ):
            return jsonify({
                "error":
                    "Image is too large. "
                    "Maximum size is 20 MB."
            }), 400

        mime_type = (
            image.mimetype
            or
            "image/jpeg"
        )

        encoded_image = (
            base64
            .b64encode(
                image_bytes
            )
            .decode(
                "utf-8"
            )
        )

        data_url = (
            f"data:{mime_type};"
            f"base64,{encoded_image}"
        )

        response = (
            client
            .chat
            .completions
            .create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {
                        "role": "system",
                        "content": VISION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        print(
            "Wiens AI vision error:",
            e
        )

        return jsonify({
            "error":
                "Image analysis error: "
                + str(e)
        }), 500


@app.route(
    "/api/transcribe",
    methods=["POST"]
)
def transcribe():
    temp_path = None

    try:
        if "audio" not in request.files:
            return jsonify({
                "error":
                    "Audio file is missing"
            }), 400

        audio = request.files[
            "audio"
        ]

        if not audio.filename:
            return jsonify({
                "error":
                    "Audio file is empty"
            }), 400

        suffix = ".webm"

        if (
            audio.filename
            .lower()
            .endswith(
                ".mp4"
            )
        ):
            suffix = ".mp4"

        elif (
            audio.filename
            .lower()
            .endswith(
                ".m4a"
            )
        ):
            suffix = ".m4a"

        elif (
            audio.filename
            .lower()
            .endswith(
                ".wav"
            )
        ):
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            audio.save(
                temp_file.name
            )

            temp_path = temp_file.name

        with open(
            temp_path,
            "rb"
        ) as audio_file:
            transcription = (
                client
                .audio
                .transcriptions
                .create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="json"
                )
            )

        text = (
            transcription
            .text
            .strip()
        )

        return jsonify({
            "text": text
        })

    except Exception as e:
        print(
            "Wiens AI transcription error:",
            e
        )

        return jsonify({
            "error":
                "Voice recognition error: "
                + str(e)
        }), 500

    finally:
        if (
            temp_path
            and
            os.path.exists(
                temp_path
            )
        ):
            try:
                os.remove(
                    temp_path
                )

            except Exception:
                pass


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
