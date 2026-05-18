# 🚀 AI Career Counsellor - Team Setup Guide

Welcome to the **AI Career Counsellor** project! This guide will walk you through exactly how to get this project running on your own machine. 

---

## 📋 Prerequisites
Before you start, make sure you have the following installed on your computer:
1. **Python 3.9+** (Make sure to check "Add Python to PATH" during installation)
2. **VS Code** (or your preferred code editor)

---

## 🛠️ Step-by-Step Setup Instructions

### Step 1: Extract the Project
1. Download the ZIP file containing the project.
2. Right-click and extract the folder to a location on your computer (e.g., `Desktop` or `Documents`).
3. Open the extracted folder in **VS Code**.

### Step 2: Open the Terminal
In VS Code, open a new terminal by clicking **Terminal -> New Terminal** at the top (or press `` Ctrl + ` ``).

### Step 3: Create a Virtual Environment
We need a clean Python environment to install the Machine Learning dependencies. Run this command in the terminal:
```bash
python -m venv venv
```
*(This will create a hidden folder called `venv`. It might take a few seconds).*

### Step 4: Activate the Virtual Environment
You must activate the environment before running anything. 

**If you are on Windows (Powershell/Command Prompt):**
```bash
venv\Scripts\activate
```
**If you are on Mac/Linux:**
```bash
source venv/bin/activate
```
*(You will know it worked if you see `(venv)` appear at the very left of your terminal prompt).*

### Step 5: Install the Dependencies
Now, install all the required AI libraries (like FastAPI, SentenceTransformers, and Scikit-Learn) by running:
```bash
pip install -r requirements.txt
```
*(This may take a few minutes as it downloads the Machine Learning tools).*

### Step 6: Start the Server!
Once everything is installed, start the backend server by running:
```bash
uvicorn backend.main:app --reload
```
Wait for the terminal to say `Application startup complete.`

---

## 🌐 Accessing the Application
Once the server is running, open your web browser (Google Chrome or Microsoft Edge recommended for voice features) and go to:

👉 **http://127.0.0.1:8000**

You can now log in, test the AI chatbot, view the live ML radar chart, and generate career reports!

---
> **Troubleshooting Tip:** If the Microphone or Text-to-Speech features don't work, ensure you are using Google Chrome and click the small microphone icon in the URL bar to grant permissions.
