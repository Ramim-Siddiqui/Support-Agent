# Support Ticket Resolution Agent (LangGraph + Gemini)

This project is a **Support Ticket Resolution Agent** built using **LangGraph** and **Google Gemini API**. It helps automate the handling and resolution of support tickets using advanced AI workflows.

---

## Setup

Follow these steps to set up the project locally.

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```
Creates a self-contained Python environment in the .venv folder.

Keeps dependencies isolated from your system Python, preventing conflicts with other projects.

### 2. Activate the Virtual Environment

**Windows:**
```bash
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

After activation, all Python packages installed will only affect this environment.

### 3. Install Project Dependencies
```bash
pip install --upgrade "langgraph-cli[inmem]"
pip install langchain
pip install google_genai
pip install langchain_chroma
pip install langchain_google_genai
```
**Explanation:**

langgraph-cli[inmem] → LangGraph CLI with in-memory backend.

langchain → Framework for building LLM-based applications.

google_genai → Google Gemini API client.

langchain_chroma → Chroma database integration for LangChain.

langchain_google_genai → LangChain support for Google Gemini.

### 4. Install the Project Locally
```bash
pip install -e .
```
Installs the project in editable mode, so changes in your code are immediately reflected without reinstalling.

### 5. Create Environment Variables File

Create a new file in the project root called .env.

Open .env and add your API keys and any other project-specific settings. For example:

GOOGLE_API_KEY=your_google_api_key_here
OTHER_SECRET_KEY=your_secret_key_here


This step ensures your sensitive credentials are stored securely and not hard-coded in the project.

### 6. Start LangGraph in Development Mode
```bash
langgraph dev
```
Launches LangGraph in development mode to test, modify, and debug your support ticket agent locally.

### Summary

By following these steps, you ensure:

A clean, isolated Python environment for your project.

All required packages for LangGraph + Gemini are installed.

Your project is ready for development and testing.