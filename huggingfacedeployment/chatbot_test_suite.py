# chatbot_test_suite.py

import json
from difflib import SequenceMatcher
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ✅ Extended test cases to cover all 13 intents
sample_test_cases = [
    # greeting
    {"input": "Hello there!", "expected_intent": "greeting", "expected_response_contains": "hello"},
    {"input": "Good morning", "expected_intent": "greeting", "expected_response_contains": "hi"},

    # farewell
    {"input": "Goodbye!", "expected_intent": "farewell", "expected_response_contains": "goodbye"},
    {"input": "Catch you later", "expected_intent": "farewell", "expected_response_contains": "see you"},

    # thanks
    {"input": "Thanks a lot!", "expected_intent": "thanks", "expected_response_contains": "welcome"},
    {"input": "Appreciate it", "expected_intent": "thanks", "expected_response_contains": "pleasure"},

    # weather
    {"input": "What's the weather in New York?", "expected_intent": "weather", "expected_entities": {"location": "New York"}, "expected_response_contains": "New York"},
    {"input": "Is it going to rain in Vancouver?", "expected_intent": "weather", "expected_entities": {"location": "Vancouver"}, "expected_response_contains": "Vancouver"},

    # product_info
    {"input": "What do you sell?", "expected_intent": "product_info", "expected_response_contains": "product"},
    {"input": "Tell me about your products", "expected_intent": "product_info", "expected_response_contains": "product"},

    # pricing
    {"input": "How much is a tablet?", "expected_intent": "pricing", "expected_entities": {"product": "tablet"}, "expected_response_contains": "tablet"},
    {"input": "What's the cost of fiction books?", "expected_intent": "pricing", "expected_entities": {"product": "fiction"}, "expected_response_contains": "fiction"},

    # scheduling
    {"input": "Schedule a meeting for tomorrow at 10am", "expected_intent": "scheduling", "expected_entities": {"date": "tomorrow", "time": "10am"}, "expected_response_contains": "tomorrow"},
    {"input": "Can I book an appointment for today at 2pm?", "expected_intent": "scheduling", "expected_entities": {"date": "today", "time": "2pm"}, "expected_response_contains": "today"},

    # yes / no
    {"input": "Yes, please go ahead", "expected_intent": "yes", "expected_response_contains": "proceed"},
    {"input": "No, not right now", "expected_intent": "no", "expected_response_contains": "no worries"},

    # order_status
    {"input": "Track my order #12345", "expected_intent": "order_status", "expected_entities": {"order_number": "12345"}, "expected_response_contains": "12345"},
    {"input": "Where is my package?", "expected_intent": "order_status", "expected_response_contains": "order"},

    # help
    {"input": "Help me with my laptop issue", "expected_intent": "help", "expected_entities": {"product": "laptop"}, "expected_response_contains": "help"},
    {"input": "I'm stuck, can you assist?", "expected_intent": "help", "expected_response_contains": "assist"},

    # get_user_info
    {"input": "Do you remember my name?", "expected_intent": "get_user_info", "expected_response_contains": "name"},
    {"input": "Who am I?", "expected_intent": "get_user_info", "expected_response_contains": "name"},

    # news
    {"input": "Tell me the news today", "expected_intent": "news", "expected_response_contains": "headlines"},
    {"input": "What's happening in the world?", "expected_intent": "news", "expected_response_contains": "news"}
]

# Utility: fuzzy match
def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Confusion matrix plot
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

# Evaluation function
def evaluate_chatbot(bot, test_cases, export_csv=False):
    results = []
    intent_matches = 0
    response_matches = 0

    predicted_intents = []
    expected_intents = []

    for case in test_cases:
        user_input = case["input"]
        expected_intent = case["expected_intent"]
        expected_phrase = case.get("expected_response_contains", "")
        expected_entities = case.get("expected_entities", {})

        predicted_intent = bot.identify_intent(user_input)
        predicted_entities = bot.extract_entities(user_input)
        actual_response = bot.process_input(user_input)

        predicted_intents.append(predicted_intent)
        expected_intents.append(expected_intent)

        # Intent check
        intent_correct = predicted_intent == expected_intent
        if intent_correct:
            intent_matches += 1

        # Response content check
        response_correct = expected_phrase.lower() in actual_response.lower()
        if response_correct:
            response_matches += 1

        # Entity F1 scoring
        entity_tp = sum(1 for k, v in expected_entities.items() if predicted_entities.get(k) == v)
        entity_fp = sum(1 for k in predicted_entities if k not in expected_entities)
        entity_fn = sum(1 for k in expected_entities if predicted_entities.get(k) != expected_entities[k])

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

    # Summary
    total = len(test_cases)
    print(f"Intent Accuracy: {intent_matches}/{total} = {intent_matches/total:.2%}")
    print(f"Response Accuracy: {response_matches}/{total} = {response_matches/total:.2%}")

    # Detailed results
    for r in results:
        print("\n[Input]", r["input"])
        print("[Intent]", "✅" if r["intent_pass"] else "❌", f"({r['actual_intent']} vs {r['expected_intent']})")
        print("[Response]", "✅" if r["response_pass"] else "❌", f"→ {r['response']}")
        print("[Entity F1]", r["f1_entity_score"])

    # Save CSV if requested
    if export_csv:
        with open("chatbot_test_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\n✅ Results saved to chatbot_test_results.csv")

    # Show confusion matrix
    plot_confusion_matrix(predicted_intents, expected_intents)

# Run evaluation
if __name__ == "__main__":
    from app import enhanced_bot
    evaluate_chatbot(enhanced_bot, sample_test_cases, export_csv=True)
