import os
import gradio as gr
import re
import random
import nltk
from datetime import datetime, timedelta
import json
import requests
from datetime import datetime, timedelta
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
import joblib

class EnhancedChatbot:
    def __init__(self):
        # Initialize intent data structure
        self.intent_model = joblib.load("intent_model.pkl")
        self.intents = {
            "greeting": {
                "patterns": [
                    "hello", "hi", "hey", "good morning", "good afternoon", 
                    "good evening", "what's up", "howdy", "greetings", "hi there"
                ],
                "responses": [
                    "Hello! How can I assist you today?",
                    "Hi there! What can I help you with?",
                    "Hey! How can I be of service?",
                    "Greetings! What do you need help with today?",
                    "Hi! How may I assist you?"
                ]
            },
            "farewell": {
                "patterns": [
                    "bye", "goodbye", "see you", "see you later", "farewell", 
                    "take care", "until next time", "catch you later", "later", "bye bye"
                ],
                "responses": [
                    "Goodbye! Have a great day!",
                    "Farewell! Feel free to chat again whenever you need assistance.",
                    "See you later! Don't hesitate to return if you have more questions.",
                    "Take care! I'll be here if you need any help in the future.",
                    "Bye for now! Come back anytime."
                ]
            },
            "thanks": {
                "patterns": [
                    "thank you", "thanks", "appreciate it", "thanks a lot", "thank you so much", 
                    "thanks for your help", "much appreciated", "that helped", "awesome, thanks", "great, thank you"
                ],
                "responses": [
                    "You're welcome!",
                    "Happy to help!",
                    "Anytime!",
                    "Glad I could assist you!",
                    "My pleasure!"
                ]
            },
            "weather": {
                "patterns": [
                    "what's the weather like", "tell me the weather", "weather forecast", 
                    "is it going to rain", "temperature today", "weather in", "how's the weather",
                    "will it be sunny", "weather conditions", "is it hot outside"
                ],
                "responses": [
                    "I'll check the weather for {location}. The current temperature is {temp}°C with {conditions}.",
                    "In {location}, it's currently {temp}°C and {conditions}.",
                    "The weather in {location} shows {temp}°C with {conditions} right now.",
                    "Looking at {location}, I see it's {temp}°C and {conditions}.",
                    "The current weather for {location} is {conditions} with a temperature of {temp}°C."
                ]
            },
            "product_info": {
                "patterns": [
                    "tell me about your products", "what do you sell", "product information", 
                    "product details", "features of product", "product specs", "tell me about product",
                    "what are your services", "available products", "product catalog"
                ],
                "responses": [
                    "We offer a variety of products including {product_type}. Would you like specific information about any of them?",
                    "Our {product_type} collection includes many options. Can I tell you about a specific product?",
                    "We specialize in high-quality {product_type}. Do you have a specific product in mind?",
                    "Our range of {product_type} is designed to meet different needs. What specific information are you looking for?",
                    "We have an extensive collection of {product_type}. Would you like me to recommend something based on your preferences?"
                ]
            },
            "scheduling": {
                "patterns": [
                    "book an appointment", "schedule a meeting", "make a reservation", 
                    "set up a call", "arrange a meeting", "book a slot", "schedule consultation",
                    "available appointment times", "when can we meet", "booking time"
                ],
                "responses": [
                    "I can help you schedule for {date} at {time}. Would you like me to confirm this appointment?",
                    "I've found an opening on {date} at {time}. Does this work for you?",
                    "We have availability on {date} at {time}. Should I book this for you?",
                    "I can schedule your appointment for {date} at {time}. Would you like to proceed?",
                    "There's an available slot on {date} at {time}. Would you like to reserve it?"
                ]
            },
            "yes": {
                "patterns": ["yes", "yeah", "yep", "sure", "of course"]
            },
            "no": {
                "patterns": ["no", "nope", "not now", "cancel"]
            },
            "order_status": {
                "patterns": [
                    "where is my order", "track my package", "order status", 
                    "when will my order arrive", "delivery status", "shipping info",
                    "check my order", "package tracking", "delivery update", "shipping status"
                ],
                "responses": [
                    "I can check the status of order #{order_number}. Currently, it's {status} and expected to arrive on {delivery_date}.",
                    "Your order #{order_number} is {status}. The estimated delivery date is {delivery_date}.",
                    "Looking at order #{order_number}, I see it's {status}. It should arrive by {delivery_date}.",
                    "The status for order #{order_number} shows {status}. Delivery is expected on {delivery_date}.",
                    "Order #{order_number} is currently {status}. You should receive it by {delivery_date}."
                ]
            },
            "help": {
                "patterns": [
                    "help me", "need assistance", "support please", "how do I", 
                    "can you help", "I'm stuck", "having trouble with", "assistance needed",
                    "support required", "instructions please"
                ],
                "responses": [
                    "I'd be happy to help with {issue}. Could you provide more details?",
                    "I can assist you with {issue}. What specifically do you need help with?",
                    "Sure, I can help with {issue}. Can you tell me more about what you're trying to do?",
                    "I'm here to help with {issue}. What seems to be the problem?",
                    "I'll help you solve your {issue}. Could you elaborate on what you're experiencing?"
                ]
            },
            "pricing": {
                "patterns": [
                    "how much does it cost", "price information", "pricing details", 
                    "what's the price", "cost of product", "rates for services", "fee structure",
                    "subscription cost", "how much for", "pricing plans"
                ],
                "responses": [
                    "The price for {product} is ${price}. Would you like more information?",
                    "{product} costs ${price}. Is there anything specific you'd like to know about it?",
                    "We offer {product} at ${price}. Would you like to hear about our other pricing options?",
                    "The current price for {product} is ${price}. Can I provide any additional details?",
                    "You can get {product} for ${price}. Are you interested in learning more about it?"
                ]
            }
        }
        
        # Initialize conversation memory
        self.conversation_memory = {
            "short_term": [],  # Stores recent messages
            "long_term": {},   # Stores persistent user information
            "current_context": None,  # Tracks current conversation context
            "pending_slots":{},
            "pending_confirmation": None,  # Stores pending confirmations
            "confirmed_appointments": []  # Stores confirmed appointments
        }
        
        # Initialize stemmer and stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Product info (example data)
        self.product_categories = ["electronics", "clothing", "furniture", "appliances", "books"]
        self.products = {
            "electronics": {"smartphone": 899, "laptop": 1299, "tablet": 499, "headphones": 199},
            "clothing": {"t-shirt": 25, "jeans": 59, "dress": 89, "jacket": 129},
            "furniture": {"sofa": 899, "table": 349, "chair": 149, "bed": 699},
            "appliances": {"refrigerator": 1099, "washing machine": 799, "microwave": 199},
            "books": {"fiction": 15, "non-fiction": 18, "cookbook": 24, "children's books": 12}
        }
        
        # Order status (example data)
        self.orders = {
            "12345": {"status": "shipped", "delivery_date": "April 5, 2025"},
            "23456": {"status": "processing", "delivery_date": "April 8, 2025"},
            "34567": {"status": "delivered", "delivery_date": "March 30, 2025"}
        }
        
        # Weather API configuration
        self.weather_api_key = "20c8b979eaab1c8afbadc75b18d06f35"  # Replace with your actual OpenWeatherMap API key
     
    def create_calendar_event(self, date, time, summary="Chatbot Appointment"):
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        service = build('calendar', 'v3', credentials=creds)

        # Create event datetime strings
        today = datetime.now()
        if date.lower() == "today":
            event_date = today
        elif date.lower() == "tomorrow":
            event_date = today + timedelta(days=1)
        elif date.lower() == "next week":
            event_date = today + timedelta(days=7)
        else:
            try:
                event_date = datetime.strptime(date, "%B %d")
                event_date = event_date.replace(year=today.year)
            except ValueError:
                return "❌ Unable to understand the date. Please use a format like 'April 5'."

        parsed_time = self.parse_natural_time(time)
        if not parsed_time:
            return "❌ Sorry, I couldn't understand the time format. Please use formats like '3pm', '14:00', or '2:15pm'."

        # Combine date and time into a full datetime
        start_datetime = datetime.combine(event_date.date(), parsed_time)
        end_datetime = start_datetime + timedelta(hours=1)

        event = {
            'summary': summary,
            'start': {'dateTime': start_datetime.isoformat(), 'timeZone': 'America/Toronto'},
            'end': {'dateTime': end_datetime.isoformat(), 'timeZone': 'America/Toronto'},
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        print(f"✅ Event created: {event.get('htmlLink')}")
        return event.get('htmlLink')
    
    def parse_natural_time(self, time_str):
        try:
            if "am" in time_str.lower() or "pm" in time_str.lower():
                return datetime.strptime(time_str.strip().lower(), "%I%p").time()
            elif ":" in time_str:
                try:
                    return datetime.strptime(time_str.strip(), "%H:%M").time()
                except ValueError:
                    return datetime.strptime(time_str.strip(), "%I:%M%p").time()
            else:
                # Assume it's a plain hour like "14"
                return datetime.strptime(time_str.strip(), "%H").time()
        except Exception:
            return None

    def preprocess_text(self, text):
        corrected_text = str(TextBlob(text).correct())
        corrected_text = corrected_text.lower()
        tokens = word_tokenize(corrected_text)
        filtered_tokens = [
            self.stemmer.stem(token) for token in tokens 
            if token not in self.stop_words and token.isalnum()
        ]
        return filtered_tokens
    
    def identify_intent(self, user_input):
        try:
            return self.intent_model.predict([user_input])[0]
        except Exception as e:
            print(f"[Intent Classification Error] {e}")
            return "unknown"

    
    def extract_entities(self, user_input):
        """Extract entities like locations, dates, names, and numbers from user input"""
        entities = {
            "location": None,
            "date": None,
            "time": None,
            "person": None,
            "number": None,
            "product": None,
            "order_number": None
        }
        
        # Extract locations
        # Simple pattern matching for "in {location}" or "for {location}"
        location_patterns = [r"in ([A-Za-z\s]+)(?:\.|\?|$)", r"for ([A-Za-z\s]+)(?:\.|\?|$)"]
        for pattern in location_patterns:
            match = re.search(pattern, user_input)
            if match:
                entities["location"] = match.group(1).strip()
                break
        
        # Extract dates
        date_patterns = [
            r"(?:on|for) ([A-Za-z]+\s\d{1,2}(?:st|nd|rd|th)?)",  # on/for January 1st
            r"(?:on|for) (\d{1,2}(?:st|nd|rd|th)?\s[A-Za-z]+)",  # on/for 1st January
            r"tomorrow",
            r"today",
            r"next week",
            r"(\d{1,2}/\d{1,2}/\d{2,4})"  # MM/DD/YYYY or DD/MM/YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                if pattern == r"tomorrow":
                    entities["date"] = "tomorrow"
                elif pattern == r"today":
                    entities["date"] = "today"
                elif pattern == r"next week":
                    entities["date"] = "next week"
                else:
                    entities["date"] = match.group(1)
                break
        
        # Extract times
        time_patterns = [
            r"at (\d{1,2}(?::\d{2})?\s?(?:am|pm))",  # at 3pm or at 3:30pm
            r"(\d{1,2}(?::\d{2})?\s?(?:am|pm))"  # 3pm or 3:30pm
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                entities["time"] = match.group(1)
                break
        
        # Extract numbers (could be order numbers, product IDs, etc.)
        number_pattern = r"#?(\d{5,})"  # Assuming order numbers are at least 5 digits
        match = re.search(number_pattern, user_input)
        if match:
            entities["order_number"] = match.group(1)
        else:
            # Look for any numbers
            match = re.search(r"(\d+)", user_input)
            if match:
                entities["number"] = match.group(1)
        
        # Extract product mentions
        for category, products in self.products.items():
            for product in products:
                if product.lower() in user_input.lower():
                    entities["product"] = product
                    break
            if entities["product"]:
                break
        
        # Use NLTK's named entity recognition for people names
        tokens = nltk.word_tokenize(user_input)
        tagged = nltk.pos_tag(tokens)
        named_entities = ne_chunk(tagged)
        
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                if chunk.label() == 'PERSON':
                    entities["person"] = ' '.join(c[0] for c in chunk)
        
        return entities
    
    def get_weather(self, location):
        """Fetch weather data from the OpenWeatherMap API with better error handling"""
        try:
            # Base URL for the OpenWeatherMap API
            base_url = "https://api.openweathermap.org/data/2.5/weather"
        
            # Parameters for the API call
            params = {
                'q': location,
                'appid': self.weather_api_key,
                'units': 'metric'  # Use Celsius
            }
            
            # Print API request details for debugging (remove in production)
            print(f"Making API request to: {base_url} with location: {location}")
            print(f"API key length: {len(self.weather_api_key)} characters")
            
            # Make the API request
            response = requests.get(base_url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant weather information
                weather_data = {
                    "temp": round(data['main']['temp']),
                    "conditions": data['weather'][0]['description'],
                    "humidity": data['main']['humidity'],
                    "wind_speed": data['wind']['speed'],
                    "city_name": data['name'],
                    "country": data['sys']['country']
                }
                
                print(f"Successfully fetched weather for {location}")
                return weather_data
            else:
                # More detailed error handling
                error_message = f"Error fetching weather data: {response.status_code}"
                if response.status_code == 401:
                    error_message += " - Unauthorized: Check your API key"
                elif response.status_code == 404:
                    error_message += f" - Location '{location}' not found"
                elif response.status_code == 429:
                    error_message += " - Too many requests: Rate limit exceeded"
                
                print(error_message)
                try:
                    error_details = response.json()
                    print(f"API Error details: {error_details}")
                except:
                    pass
                
                # Return dummy data with an indication it's not real
                return {
                    "temp": 20, 
                    "conditions": "cloudy (simulated data)",
                    "humidity": 70, 
                    "wind_speed": 5,
                    "city_name": location,
                    "note": "Using simulated weather data due to API error"
                }
        
        except Exception as e:
            # Handle any exceptions that might occur
            print(f"Exception in weather API call: {str(e)}")
            return {
                "temp": 20, 
                "conditions": "cloudy (simulated data)",
                "humidity": 70, 
                "wind_speed": 5,
                "city_name": location,
                "note": "Using simulated weather data due to exception"
            }
            
    def generate_response(self, intent, entities):
        """Generate a response based on the identified intent and extracted entities"""
        if intent == "unknown":
            return "I'm not sure I understand. Could you rephrase that or provide more details?"
        
        # Select a random response template for the intent
        response_template = random.choice(self.intents[intent]["responses"])
        
        # Fill in the template with entity information
        if intent == "weather":
            location = entities["location"] or "your location"
            
            # Handle case where location is "your location"
            if location == "your location" and "preferred_location" in self.conversation_memory["long_term"]:
                location = self.conversation_memory["long_term"]["preferred_location"]
            elif location == "your location":
                return "Could you specify which location you'd like the weather for?"
            
            weather_data = self.get_weather(location)
            
            response = response_template.format(
                location=weather_data.get("city_name", location),
                temp=weather_data["temp"],
                conditions=weather_data["conditions"]
            )
            
            # Add extra weather information
            if "humidity" in weather_data and "wind_speed" in weather_data:
                response += f" Humidity is {weather_data['humidity']}% with wind speed of {weather_data['wind_speed']} m/s."
        
        elif intent == "product_info":
            product_type = random.choice(self.product_categories)
            response = response_template.format(product_type=product_type)
        
        elif intent == "scheduling":
            date = entities.get("date")
            time = entities.get("time")

            if not date or not time:
            # Save the pending info
                self.conversation_memory["pending_slots"] = {
                    "intent": intent,
                    "required": [],
                    "filled": {}
                }
                if not date:
                    self.conversation_memory["pending_slots"]["required"].append("date")
                else:
                    self.conversation_memory["pending_slots"]["filled"]["date"] = date

                if not time:
                    self.conversation_memory["pending_slots"]["required"].append("time")
                else:
                    self.conversation_memory["pending_slots"]["filled"]["time"] = time

                # Ask for missing info
                if not date:
                    return "Sure! What date would you like to book the appointment for?"
                if not time:
                    return "Great. What time would work best for your appointment?"

            # All slots filled
            self.conversation_memory["pending_confirmation"] = {
            "date": date,
            "time": time
             }
            response_template = random.choice(self.intents[intent]["responses"])
            return response_template.format(date=date, time=time)
        elif intent == "order_status":
            order_number = entities["order_number"] or "N/A"
            if order_number in self.orders:
                order_info = self.orders[order_number]
                response = response_template.format(
                    order_number=order_number,
                    status=order_info["status"],
                    delivery_date=order_info["delivery_date"]
                )
            else:
                response = "I couldn't find an order with that number. Could you verify the order number and try again?"
        
        elif intent == "help":
            issue = entities["product"] or "your issue"
            response = response_template.format(issue=issue)
        
        elif intent == "pricing":
            product = entities["product"]
            price = "N/A"
            
            if product:
                # Search for the product in our product database
                for category, products in self.products.items():
                    if product in products:
                        price = products[product]
                        break
                
                response = response_template.format(product=product, price=price)
            else:
                response = "What specific product would you like pricing information for?"
        
        else:
            # For intents without entity-dependent responses (greeting, farewell, thanks)
            response = response_template
        
        return response
    
    def update_conversation_memory(self, user_input, intent, entities, response):
        """Update the conversation memory with the current exchange"""
        # Update short-term memory (last 5 exchanges)
        self.conversation_memory["short_term"].append({
            "user_input": user_input,
            "intent": intent,
            "entities": entities,
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Keep only the last 5 exchanges
        if len(self.conversation_memory["short_term"]) > 5:
            self.conversation_memory["short_term"] = self.conversation_memory["short_term"][-5:]
        
        # Update long-term memory with persistent information
        if entities["location"] and entities["location"] != "your location":
            self.conversation_memory["long_term"]["preferred_location"] = entities["location"]
        
        if entities["person"]:
            self.conversation_memory["long_term"]["user_name"] = entities["person"]
        
        # Update current context
        self.conversation_memory["current_context"] = intent
    
    def personalize_response(self, response):
        """Add personalization to the response based on conversation memory"""
        # If we know the user's name, occasionally use it
        if "user_name" in self.conversation_memory["long_term"] and random.random() < 0.8:
            # Only add the name if it's not already in the response
            user_name = self.conversation_memory["long_term"]["user_name"]
            if user_name not in response:
                response = f"{user_name}, {response[0].lower() + response[1:]}"
        
        return response
    
    def check_conversation_flow(self, intent):
        """Check if the current intent makes sense given the conversation history"""
        if not self.conversation_memory["short_term"]:
            return True
        
        # Get the previous intent
        previous_intent = self.conversation_memory["short_term"][-1]["intent"]
        
        # Define logical conversation flows
        logical_flows = {
            "greeting": ["weather", "product_info", "scheduling", "order_status", "help", "pricing"],
            "product_info": ["pricing", "order_status", "scheduling", "farewell"],
            "pricing": ["product_info", "scheduling", "farewell"],
            "order_status": ["help", "farewell"],
            "scheduling": ["product_info", "farewell"],
            "weather": ["scheduling", "product_info", "farewell"],
            "help": ["product_info", "pricing", "scheduling", "order_status", "farewell"],
            "thanks": ["farewell"],
            "farewell": ["greeting"]
        }
        
        # Check if the current intent follows logically from the previous one
        if previous_intent in logical_flows and intent in logical_flows[previous_intent]:
            return True
        
        # If not a direct logical flow, it's still acceptable but might require a transition
        return False
    
    def process_input(self, user_input):
        # ✅ Step 0: Handle confirmation
        if self.conversation_memory.get("pending_confirmation"):
            intent = self.identify_intent(user_input)
            if intent == "yes":
                confirmed = self.conversation_memory["pending_confirmation"]
                self.conversation_memory.setdefault("confirmed_appointments", []).append(confirmed)
                self.conversation_memory["pending_confirmation"] = None
                link = self.create_calendar_event(confirmed["date"], confirmed["time"])
                return f"✅ Your appointment has been confirmed for {confirmed['date']} at {confirmed['time']}.\n📅 [View in Google Calendar]({link})"
            elif intent == "no":
                self.conversation_memory["pending_confirmation"] = None
                return "Okay, I won't schedule it. Let me know if you'd like to pick a new time."

        # ✅ Step 1: Handle slot filling
        if self.conversation_memory.get("pending_slots"):
            pending = self.conversation_memory["pending_slots"]
            updated_entities = self.extract_entities(user_input)

            for slot in pending["required"]:
                if updated_entities.get(slot):
                    pending["filled"][slot] = updated_entities[slot]

            if all(slot in pending["filled"] for slot in pending["required"]):
                intent = pending["intent"]
                date = pending["filled"].get("date")
                time = pending["filled"].get("time")

                # ✅ Check for conflict
                existing = self.conversation_memory.get("confirmed_appointments", [])
                conflict = any(appt["date"] == date and appt["time"] == time for appt in existing)

                if conflict:
                    self.conversation_memory["pending_slots"]["filled"].pop("time", None)
                    return f"⚠️ Sorry, an appointment is already scheduled for {date} at {time}. Please choose a different time."

                # No conflict – proceed to confirmation
                response_template = random.choice(self.intents[intent]["responses"])
                self.conversation_memory["pending_slots"] = {}
                self.conversation_memory["pending_confirmation"] = {"date": date, "time": time}
                response = response_template.format(date=date, time=time)
                self.update_conversation_memory(user_input, intent, updated_entities, response)
                return self.personalize_response(response)

            # Ask for missing fields
            if "date" in pending["required"] and "date" not in pending["filled"]:
                return "Got it. What date would you prefer?"
            if "time" in pending["required"] and "time" not in pending["filled"]:
                return "And what time would be good for the appointment?"

        # ✅ Step 2: Detect intent
        print("\n--- BEFORE PROCESSING ---")
        print(f"Input: '{user_input}'")
        print(f"Current Memory: {json.dumps(self.conversation_memory, indent=2, default=str)}")

        intent = self.identify_intent(user_input)
        print(f"\n--- AFTER INTENT DETECTION ---")
        print(f"Detected Intent: {intent}")

        entities = self.extract_entities(user_input)

        # ⬇️ Fallback: if intent is unknown but we have a previous context
        if intent == "unknown" and self.conversation_memory.get("current_context") in ["weather", "scheduling", "order_status"]:
            print("ℹ️ Falling back to previous intent:", self.conversation_memory["current_context"])
            intent = self.conversation_memory["current_context"]

        # ⬇️ If no location found, try treating input as one-word location
        if intent == "weather" and not entities.get("location"):
            if len(user_input.split()) == 1 and user_input.isalpha():
                entities["location"] = user_input.strip()

        print(f"\n--- AFTER ENTITY EXTRACTION ---")
        print(f"Extracted Entities: {json.dumps(entities, indent=2)}")

        flow_is_logical = self.check_conversation_flow(intent)
        print(f"\n--- CONVERSATION FLOW CHECK ---")
        print(f"Flow is logical: {flow_is_logical}")

        response = self.generate_response(intent, entities)
        print(f"\n--- AFTER RESPONSE GENERATION ---")
        print(f"Generated Response: '{response}'")

        if not flow_is_logical and intent != "unknown":
            transitions = {
                "greeting": "Hello there! ",
                "product_info": "About our products, ",
                "pricing": "Regarding pricing, ",
                "order_status": "About your order, ",
                "scheduling": "About scheduling, ",
                "weather": "Checking the weather, ",
                "help": "To help you, ",
                "thanks": "You're welcome! ",
                "farewell": "Before you go, "
            }
            response = transitions.get(intent, "") + response
            print(f"\n--- AFTER ADDING TRANSITION ---")
            print(f"Response with transition: '{response}'")

        personalized_response = self.personalize_response(response)
        if personalized_response != response:
            print(f"\n--- AFTER PERSONALIZATION ---")
            print(f"Personalized Response: '{personalized_response}'")

        self.update_conversation_memory(user_input, intent, entities, personalized_response)
        print(f"\n--- AFTER MEMORY UPDATE ---")
        print(f"Updated Memory: {json.dumps(self.conversation_memory, indent=2, default=str)}")

        return personalized_response




# Initialize the chatbot instance
enhanced_bot = EnhancedChatbot()

# Function to handle chat interactions for Gradio
def respond(message, history):
    # Use the EnhancedChatbot instance to process the input
    bot_message = enhanced_bot.process_input(message)
    
    # Update memory state after each message
    # This will be called automatically when respond is used
    return bot_message

# Function to get current memory state
def get_memory_state():
    return enhanced_bot.conversation_memory

# Function to debug both intent/entities and memory
def debug_response(message):
    intent = enhanced_bot.identify_intent(message)
    entities = enhanced_bot.extract_entities(message)
    memory_info = json.dumps(enhanced_bot.conversation_memory, indent=2, default=str)
    return f"Intent: {intent}\nEntities: {json.dumps(entities, indent=2)}\nMemory: {memory_info}"

# Create Gradio interface with a modern theme
# Launch UI for Enhanced Chatbot with Custom Styling
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container { font-family: 'Segoe UI', sans-serif; }
    .chatbot .message.bot { background-color: #eef3f9; }
    .chatbot .message.user { background-color: #dcfce7; }
    .gr-button { border-radius: 10px !important; }
    .gr-textbox { border-radius: 10px !important; }
""") as demo:

    # Header
    gr.Markdown("""
    # 🤖 Smart Assistant  
    Welcome to your AI-powered assistant. Ask about **weather**, **products**, **appointments**, or **order tracking**.
    """)

    # Chatbot Display
    chatbot = gr.Chatbot(
        height=450,
        bubble_full_width=False,
        label="💬 Assistant",
        avatar_images=("avatar.png", None)  # Optional: replace with your icons
    )

    # Textbox + Buttons in a row
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            container=False,
            scale=6
        )
        submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)
        clear_btn = gr.Button("🗑️ Clear", scale=1)

    # Conversation Memory Viewer
    with gr.Accordion("🧠 Conversation Memory", open=False):
        memory_output = gr.JSON(label="📦 Current Memory State")
        refresh_button = gr.Button("🔄 Refresh Memory")

    # Debugging & Development Tools
    with gr.Accordion("🛠️ Developer Tools", open=False):
        gr.Markdown("#### 🔍 Intent & Entity Debugger")
        with gr.Row():
            debug_input = gr.Textbox(label="Test Message")
            debug_output = gr.Textbox(label="Analysis Result", lines=10, interactive=False)
        debug_button = gr.Button("🧪 Analyze")

        gr.Markdown("#### 💡 Example Inputs")
        gr.Examples(
            examples=[
                "What's the weather like in New York?",
                "Tell me about your smartphone products",
                "How much does a laptop cost?",
                "Can I schedule an appointment for tomorrow at 3pm?",
                "Where is my order #12345?",
                "I need help with my tablet",
                "My name is Sarah and I live in Boston",
                "What's the weather like today?"
            ],
            inputs=debug_input
        )

    # Functions
    def user_input(user_message, history):
        return "", history + [[user_message, None]]

    def bot_response(history):
        if history and history[-1][1] is None:
            bot_message = enhanced_bot.process_input(history[-1][0])
            history[-1][1] = bot_message
        return history, enhanced_bot.conversation_memory

    def clear_chat_history():
        return [], enhanced_bot.conversation_memory

    # Button Logic
    msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot], [chatbot, memory_output]
    )

    submit_btn.click(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot], [chatbot, memory_output]
    )

    clear_btn.click(clear_chat_history, [], [chatbot, memory_output], queue=False)
    refresh_button.click(get_memory_state, None, memory_output)
    debug_button.click(debug_response, inputs=debug_input, outputs=debug_output)

# Launch the app
if __name__ == "__main__":
    demo.launch()
