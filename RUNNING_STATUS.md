# 🚀 Career Counsellor - Running Status & Guide

The **AI Career Counsellor** project is **up and running successfully** on your local machine! 

---

## 🟢 Live Instance Details

* **Local URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **API Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
* **Status:** `Healthy` 🟢
* **Service:** `AI Career Counsellor Chatbot`

---

## 🗄️ Database & Pre-registered Users

We verified the local SQLite database (`career_bot.db`) and found it is fully initialized with the following tables:
* `users`
* `sessions`
* `chat_history`
* `results`

### 👤 Existing Credentials
There is already a registered user account you can use to sign in immediately:
* **Username:** `hem`
* **Registered Email:** `hemgabhawala@gmail.com`
* **Full Name:** `Hem Nikesh Gabhawala`

*Alternatively, you can click the **Register** tab on the login screen to create a new student account.*

---

## 🛠️ Commands for Future Reference

If you ever need to stop, restart, or run the server manually in a new terminal, use the following steps:

### 1. Navigate to the Project Directory
```bash
cd /Users/hemgabhawala/Desktop/AI_Final_Hackathon/Career_Counsellor
```

### 2. Run the FastAPI Server (with correct PYTHONPATH)
```bash
PYTHONPATH=backend ./venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

> [!TIP]
> **Recommended Browser:** Google Chrome or Microsoft Edge.
> If you plan to test the live voice synthesis (text-to-speech) features, click the megaphone icon `🔊` in the input panel to activate speech output.
