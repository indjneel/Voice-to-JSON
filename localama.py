import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
import whisper
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "SimplyParseChatbot"

GITHUB_TOKEN = "token"
GITHUB_REPO = "indjneel/simply-chatbot"

def create_issue(title, body):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(" issue created ")
    else:
        print(f" failed to create issue: {response.status_code} - {response.text}")

whisper_model = whisper.load_model("base")

llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            """You are a knowledgeable and helpful virtual assistant specialized in **document parsing, troubleshooting, and help desk support**.

            Your primary roles are:
        1. **Document Parsing Guidance:** Help users understand how to upload, highlight, and parse documents (e.g., invoices, receipts, PDFs, Word/Excel files) to extract structured data.
        2. **Troubleshooting:** Assist users in resolving common errors, OCR issues, parsing inaccuracies, or software-related problems.
        3. **Help Desk Operations:** Provide guidance on general software features, explain workflows, and offer tips for efficient document management.

        **Behavior and Guidelines:**
        - Always respond in a **clear, concise, and step-by-step manner**.
        - Use **friendly and professional language**, but avoid unnecessary jargon.
        - Ask **clarifying questions** if the user's issue is unclear before giving a solution.
        - Break instructions into **bullet points or numbered steps** for clarity.
        - Provide **examples or code blocks** when explaining technical steps.
        - Give **JSON or structured data examples** only when relevant to parsing tasks.
        - Suggest **workarounds or alternatives** if a direct solution is not possible.

        **Response Formatting:**
        - Use headings like `Document Parsing`, `Troubleshooting`, or `Help Desk` when relevant.
        - Make your answers **actionable and precise**, avoiding long paragraphs unless necessary.
        - Keep instructions general so they can be applied in a variety of software environments.

        **Example Prompts You Should Handle:**
        - "How do I extract data from a scanned invoice?"
        - "The OCR output is incorrect, what should I do?"
        - "How do I upload multiple documents and parse them together?"
        - "Can you give me a step-by-step guide for configuring automated parsing rules?"
        """
         ),
    ]
)    

def create_github_issue(title, body):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")  # format: "username/repo"
    url = f"https://api.github.com/repos/{repo}/issues"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    data = {
        "title": title,
        "body": body
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        return response.json()["html_url"]
    else:
        return f"failed to create issue: {response.status_code} - {response.text}"

st.title("Simar - Assistant")

st.subheader(" demo video")
video_file_path = r"Add path here"
try:
    video_file = open(video_file_path, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
except Exception:
    st.warning("demo video file not found. Please check the path.")

# Category selection
category = st.radio("Select a category", ["Issue", "Help", "Doubts"])

if category == "Help":
    st.markdown("[Click here for Help Center](add your email)")

elif category == "Issue":
    st.subheader("Submit an Issue to GitHub")
    issue_title = st.text_input("Issue Title")
    issue_body = st.text_area("Issue Description")

    if st.button("Submit Issue to GitHub"):
        if issue_title.strip() and issue_body.strip():
            with st.spinner("Creating GitHub issue..."):
                result = create_github_issue(issue_title.strip(), issue_body.strip())
            if result.startswith("http"):
                st.success(f"Issue created successfully! [View Issue here]({result})")
            else:
                st.error(result)
        else:
            st.error("Please fill in both the issue title and description.")

elif category == "Doubts":
    st.subheader("Ask your questions")

    input_text = st.text_input("Ask something")

    audio_file = st.file_uploader("Upload audio note for transcription and JSON parsing", type=["wav", "mp3", "m4a"])

    if input_text:
        chain = prompt | llm | output_parser
        response = chain.invoke({"question": input_text})
        st.markdown("###  Simar's Response")
        st.write(response)

    if audio_file:
        st.markdown("###  Transcribing audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        transcription = whisper_model.transcribe(tmp_path)
        transcribed_text = transcription["text"]
        st.markdown("###  Transcription")
        st.success(transcribed_text)

        json_prompt = f"""
You are a financial assistant. Based on the user message below, extract all possible structured data in JSON format.
If any field cannot be found, fill it with null.

Message: \"{transcribed_text}\"

Extract JSON with this structure:
{{
  "invoice_number": null,
  "company_name": null,
  "invoice_date": null,
  "due_date": null,
  "total_amount": null,
  "currency": null,
  "items": [
    {{
      "description": null,
      "quantity": null,
      "unit_price": null,
      "total_price": null
    }}
  ],
  "notes": null
}}
Return ONLY the JSON.
"""
        st.markdown("###  Sending to local LLM (Ollama)...")
        try:
            response = llm.invoke(json_prompt)
            parsed_json = json.loads(response)
            st.markdown("###  Parsed JSON Output")
            st.json(parsed_json)
        except Exception as e:
            st.error(" Failed to parse JSON from Ollama response.")
            st.text("Raw LLM Output:")
            st.code(response if 'response' in locals() else str(e))
