
# 🤖 Smart Bot: Weather & Scheduling Assistant

An intelligent AI chatbot assistant built with **Gradio**, designed to help users with:

- 🌤️ Weather queries  
- 📅 Appointment scheduling with **Google Calendar integration**  
- 📦 Order tracking  
- 🛍️ Product info & pricing  
- 🧠 Memory-aware conversations with context tracking  
- 📖 Paragraph-based QA using transformers

---

## 🚀 Features

| Feature              | Description |
|----------------------|-------------|
| **Natural Chat Interface** | Built using Gradio with a clean UI, typing responses, and example prompts |
| **Intent Recognition**     | Trained ML model using `TfidfVectorizer` + `LogisticRegression` (via `joblib`) |
| **Entity Extraction**      | Extracts date, time, location, name, order number, etc. using NLTK & regex |
| **Slot Filling**           | Remembers partial inputs and follows up for missing info |
| **Context Tracking**       | Maintains conversation flow and reuses past information (e.g., names, locations) |
| **Appointment Booking**    | Integrated with Google Calendar using OAuth & Google API |
| **Rescheduling & Cancelling** | Users can reschedule or cancel appointments by natural language |
| **Weather API**            | Uses OpenWeatherMap to fetch live weather |
| **Paragraph QA**           | Users can paste a paragraph and ask questions based on it |
| **Developer Tools**        | Debug intent and memory directly from the UI |

---

## 🧠 How It Works

1. **Intent Detection**: Uses a trained ML model (intent_model.pkl) to classify user input.
2. **Entity Extraction**: Parses inputs for useful info like name, date, time, etc.
3. **Memory Management**: Saves short-term conversation history and long-term facts (like user's name or preferred location).
4. **API Integration**:
   - **Weather**: OpenWeatherMap
   - **Calendar**: Google Calendar (with OAuth flow)
   - **QA**: Transformers (DistilBERT)

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Smart-Bot-Weather---Scheduling.git
cd Smart-Bot-Weather---Scheduling
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your credentials

> **Important:** `credentials.json` and `token.json` should NOT be committed to GitHub. Add them to `.gitignore`.

- Create a Google Cloud project and download your `credentials.json`
- It will be used during the first OAuth login to create `token.json`

### 5. Train the Intent Model (Optional)

If you want to retrain the intent classifier:

```bash
python intent_trainer.py
```

---

## ▶️ Running the App

```bash
python app.py
```

> The app will open in your browser at `http://127.0.0.1:7860`

---

## 📦 Deployment

To deploy on Hugging Face Spaces or similar:

- Use `gr.Interface` or `gr.Blocks`
- Add `requirements.txt` and `README.md`
- Upload to your Hugging Face repo

---

## 🧪 Sample Inputs

```plaintext
"What's the weather in Toronto?"
"Book an appointment for tomorrow at 2pm"
"Where is my order #12345?"
"My name is Sarah"
"Can I reschedule my appointment to Monday at 4pm?"
"Cancel my meeting for tomorrow"
```

---

## 🔐 Security Note

Ensure you **do NOT commit secrets** (like Google API credentials) to GitHub.  
Use `.gitignore` to exclude sensitive files:

```gitignore
token.json
credentials.json
```

---

## 📄 License

MIT License. Feel free to use, modify, and share.

## 👥 Collaborators

- [Kiran Gobi Manivannan](https://github.com/Kiran14082000)
- [Gowtham SR](https://github.com/Gowtham-sr)
- [Skowshi](https://github.com/skowshi)

