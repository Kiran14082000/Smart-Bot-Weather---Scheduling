# intent_trainer.py

import json
from difflib import SequenceMatcher
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# --- TRAIN INTENT CLASSIFIER ---

def load_training_data():
    # Sample training data extracted from intents
    return [
        {"text": "Hi there!", "intent": "greeting"},
        {"text": "Hello!", "intent": "greeting"},
        {"text": "Good morning", "intent": "greeting"},
        {"text": "Bye!", "intent": "farewell"},
        {"text": "See you later", "intent": "farewell"},
        {"text": "Thanks a lot", "intent": "thanks"},
        {"text": "Thank you!", "intent": "thanks"},
        {"text": "What's the weather in Toronto?", "intent": "weather"},
        {"text": "Will it rain tomorrow?", "intent": "weather"},
        {"text": "How much is a laptop?", "intent": "pricing"},
        {"text": "Book me for tomorrow at 2pm", "intent": "scheduling"},
        {"text": "Can I make an appointment?", "intent": "scheduling"},
        {"text": "Track order #12345", "intent": "order_status"},
        {"text": "Where is my package?", "intent": "order_status"},
        {"text": "Help me with my tablet", "intent": "help"}
    ]


def train_intent_classifier():
    data = load_training_data()
    texts = [item["text"] for item in data]
    labels = [item["intent"] for item in data]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(texts, labels)
    joblib.dump(pipeline, "intent_model.pkl")
    print("✅ Intent model trained and saved as intent_model.pkl")


# --- CHATBOT EVALUATION ---

sample_test_cases = [
    {"input": "Hi there!", "expected_intent": "greeting", "expected_response_contains": "hello"},
    {"input": "What's the weather in Toronto?", "expected_intent": "weather", "expected_entities": {"location": "Toronto"}, "expected_response_contains": "Toronto"},
    {"input": "How much is a laptop?", "expected_intent": "pricing", "expected_response_contains": "laptop"},
    {"input": "Can I book an appointment for tomorrow at 3pm?", "expected_intent": "scheduling", "expected_entities": {"date": "tomorrow", "time": "3pm"}, "expected_response_contains": "tomorrow"}
]


def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def plot_confusion_matrix(predicted, expected):
    labels = sorted(set(predicted + expected))
    matrix = [[0 for _ in labels] for _ in labels]
    label_index = {label: idx for idx, label in enumerate(labels)}

    for p, e in zip(predicted, expected):
        matrix[label_index[e]][label_index[p]] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.title("Intent Confusion Matrix")
    plt.show()


def evaluate_chatbot(bot, test_cases, export_csv=False):
    results = []
    intent_matches = 0
    response_matches = 0

    predicted_intents = []
    expected_intents = []

    for i, case in enumerate(test_cases):
        user_input = case["input"]
        expected_intent = case["expected_intent"]
        expected_phrase = case.get("expected_response_contains", "")
        expected_entities = case.get("expected_entities", {})

        predicted_intent = bot.identify_intent(user_input)
        predicted_entities = bot.extract_entities(user_input)
        actual_response = bot.process_input(user_input)

        predicted_intents.append(predicted_intent)
        expected_intents.append(expected_intent)

        intent_correct = predicted_intent == expected_intent
        if intent_correct:
            intent_matches += 1

        response_correct = expected_phrase.lower() in actual_response.lower()
        if response_correct:
            response_matches += 1

        entity_tp = 0
        entity_fp = 0
        entity_fn = 0

        for k, v in expected_entities.items():
            if predicted_entities.get(k) == v:
                entity_tp += 1
            else:
                entity_fn += 1

        for k, v in predicted_entities.items():
            if k not in expected_entities:
                entity_fp += 1

        precision = entity_tp / (entity_tp + entity_fp + 1e-6)
        recall = entity_tp / (entity_tp + entity_fn + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)

        results.append({
            "input": user_input,
            "intent_pass": intent_correct,
            "response_pass": response_correct,
            "actual_intent": predicted_intent,
            "expected_intent": expected_intent,
            "response": actual_response,
            "f1_entity_score": round(f1_score, 2)
        })

    total = len(test_cases)
    print(f"Intent Accuracy: {intent_matches}/{total} = {intent_matches/total:.2%}")
    print(f"Response Accuracy: {response_matches}/{total} = {response_matches/total:.2%}")

    for r in results:
        print("\n[Input]", r["input"])
        print("[Intent]", "✅" if r["intent_pass"] else "❌", f"({r['actual_intent']} vs {r['expected_intent']})")
        print("[Response]", "✅" if r["response_pass"] else "❌", f"→ {r['response']}")
        print("[Entity F1]", r["f1_entity_score"])

    if export_csv:
        with open("chatbot_test_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\n✅ Results saved to chatbot_test_results.csv")

    plot_confusion_matrix(predicted_intents, expected_intents)


if __name__ == "__main__":
    train_intent_classifier()
    from app import enhanced_bot
    evaluate_chatbot(enhanced_bot, sample_test_cases, export_csv=True)
