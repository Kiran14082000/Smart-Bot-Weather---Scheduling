import joblib
import re
import random
import nltk
import json
import requests
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from textblob import TextBlob
import gradio as gr

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class EnhancedChatbot:
    def __init__(self):
        self.intent_model = joblib.load("intent_model.pkl")

        self.intents = {
            "greeting": {"responses": ["Hello! How can I assist you today?", "Hi there!", "Hey!"]},
            "farewell": {"responses": ["Goodbye! Have a great day!", "Farewell!", "See you later!"]},
            "thanks": {"responses": ["You're welcome!", "Anytime!", "Glad I could help!"]},
            "weather": {"responses": ["In {location}, it's {temp}°C with {conditions}."]},
            "product_info": {"responses": ["We have a range of {product_type}. Do you want details on something specific?"]},
            "scheduling": {"responses": ["Scheduled on {date} at {time}. Confirm?"]},
            "yes": {"responses": ["Great! Proceeding."]},
            "no": {"responses": ["No worries. Let me know if you need anything."]},
            "order_status": {"responses": ["Order #{order_number} is {status}. Expected on {delivery_date}."]},
            "help": {"responses": ["I can assist with {issue}. Could you elaborate?"]},
            "pricing": {"responses": ["The price for {product} is ${price}. Want more info?"]},
            "get_user_info": {"responses": ["Your name is {name}."]},
            "news": {"responses": ["Top headlines:\n{news_list}"]}
        }

        self.products = {
            "electronics": {"laptop": 1200, "tablet": 500},
            "books": {"fiction": 20, "non-fiction": 25}
        }

        self.orders = {
            "12345": {"status": "shipped", "delivery_date": "April 15, 2025"}
        }

        self.weather_api_key = "20c8b979eaab1c8afbadc75b18d06f35"  # ✅ Replace with your real key if needed
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

        self.memory = {
            "long_term": {},
            "short_term": []
        }

    def preprocess_text(self, text):
        corrected = str(TextBlob(text).correct()).lower()
        tokens = word_tokenize(corrected)
        return [self.stemmer.stem(t) for t in tokens if t not in self.stop_words and t.isalnum()]

    def identify_intent(self, text):
        try:
            return self.intent_model.predict([text])[0]
        except:
            return "unknown"

    def extract_entities(self, text):
        entities = {
            "location": None, "product": None, "order_number": None,
            "person": None, "date": None, "time": None
        }

        loc_match = re.search(r"in ([A-Za-z\s]+)", text)
        if loc_match:
            entities["location"] = loc_match.group(1).strip()

        if "tomorrow" in text.lower():
            entities["date"] = "tomorrow"
        if "today" in text.lower():
            entities["date"] = "today"
        time_match = re.search(r"(\d{1,2} ?(am|pm))", text.lower())
        if time_match:
            entities["time"] = time_match.group(1)

        order_match = re.search(r"#?(\d{5,})", text)
        if order_match:
            entities["order_number"] = order_match.group(1)

        for cat, items in self.products.items():
            for product in items:
                if product in text.lower():
                    entities["product"] = product

        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        named_entities = ne_chunk(tagged)
        for chunk in named_entities:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                entities["person"] = ' '.join(c[0] for c in chunk)

        return entities

    def get_weather(self, location):
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": self.weather_api_key, "units": "metric"}
        try:
            res = requests.get(url, params=params)
            data = res.json()

            if res.status_code != 200:
                raise ValueError(data.get("message", "Unknown error"))

            return {
                "location": data.get("name", location),
                "temp": round(data['main']['temp']),
                "conditions": data['weather'][0]['description']
            }
        except Exception as e:
            print(f"⚠️ Weather fetch failed: {e}")
            return {
                "location": location,
                "temp": 20,
                "conditions": "cloudy (fallback due to error)"
            }

    def get_latest_news(self):
        return "\n".join([
            "1. AI revolutionizes education.",
            "2. Tech giants invest in climate solutions.",
            "3. Breakthroughs in cancer research."
        ])

    def generate_response(self, intent, entities):
        template = random.choice(self.intents[intent]["responses"])

        if intent == "weather":
            location = entities["location"] or "your location"
            weather = self.get_weather(location)
            return template.format(**weather)

        if intent == "product_info":
            return template.format(product_type="electronics")

        if intent == "pricing":
            product = entities.get("product")
            price = None
            for cat in self.products:
                if product in self.products[cat]:
                    price = self.products[cat][product]
            return template.format(product=product or "that item", price=price or "N/A")

        if intent == "order_status":
            order = self.orders.get(entities.get("order_number"))
            if order:
                return template.format(order_number=entities["order_number"], **order)
            else:
                return "I couldn't find that order. Can you double-check the number?"

        if intent == "help":
            return template.format(issue=entities.get("product") or "your issue")

        if intent == "get_user_info":
            name = entities.get("person") or self.memory["long_term"].get("user_name", "friend")
            return template.format(name=name)

        if intent == "news":
            return template.format(news_list=self.get_latest_news())

        if intent == "scheduling":
            if not entities.get("date") or not entities.get("time"):
                return "Please specify both date and time."
            return template.format(date=entities["date"], time=entities["time"])

        return template

    def process_input(self, user_input):
        intent = self.identify_intent(user_input)
        entities = self.extract_entities(user_input)
        response = self.generate_response(intent, entities)

        if entities.get("person"):
            self.memory["long_term"]["user_name"] = entities["person"]

        return response


# Gradio Interface
bot = EnhancedChatbot()

def chat(message, history):
    return bot.process_input(message)

demo = gr.ChatInterface(chat, title="🧠 AI Assistant", theme="soft")

if __name__ == "__main__":
    demo.launch()
