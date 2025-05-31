from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from phi.tools.youtube_tools import YouTubeTools

load_dotenv()

app = Flask(__name__)
CORS(app)

# Setup Groq Agent
model = Groq(
    id="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY")
)

agent = Agent(
    model=model,
    tools=[YouTubeTools()],
    show_tool_calls=True,
    description="You are a YouTube/video analysis agent. Generate summaries from links or uploads.",
)

@app.route("/api/summarize", methods=["POST"])
def summarize_youtube_video():
    data = request.json
    link = data.get("link")
    if not link:
        return jsonify({"error": "Missing YouTube link"}), 400

    prompt = f"""
    You are given a YouTube video link: {link}

    Your task is to:
    1. Extract the **title**, **author**, and **thumbnail**.
    2. Fetch captions or transcript (if available).
    3. Generate a **detailed summary**:
       - What is the video about?
       - Key points and topics discussed.
       - Intended audience and tone.

    Format output in clear **Markdown**.
    """

    output = agent.run(prompt, markdown=True)
    return jsonify({"summary": output.content})

if __name__ == "__main__":
    app.run(debug=True)
