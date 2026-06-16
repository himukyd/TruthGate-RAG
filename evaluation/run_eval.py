import os
import sys
import re
import json
import time
from collections import defaultdict

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import List
from rag.graph import process_query
from app.schemas import QueryResponse


def sanitize_filename(text: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z\-_. ]+", "", text)
    safe = safe.strip().replace(" ", "_")
    return safe[:80] or "question"


def save_question_response(index: int, category: str, query: str, expected_status: str, response: QueryResponse, output_dir="evaluation/qa_responses"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{index:03d}_{sanitize_filename(query)}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Question: {query}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Expected Status: {expected_status}\n")
        f.write(f"Actual Status: {response.status}\n")
        f.write(f"Latency: {response.latency:.4f}s\n")
        f.write("\nAnswer:\n")
        f.write(response.answer + "\n")


def load_questions(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_question_limit() -> int:
    try:
        return int(os.getenv("QUESTION_LIMIT", "20"))
    except ValueError:
        return 20


def sample_questions_by_category(questions: List[dict], limit: int) -> List[dict]:
    if limit >= len(questions):
        return questions

    categories = defaultdict(list)
    for item in questions:
        categories[item["category"]].append(item)

    category_order = sorted(categories.keys())
    per_category = max(1, limit // len(category_order))

    selected = []
    for category in category_order:
        selected.extend(categories[category][:per_category])

    remaining = limit - len(selected)
    if remaining > 0:
        for category in category_order:
            if remaining <= 0:
                break
            extra = categories[category][per_category:per_category + remaining]
            selected.extend(extra)
            remaining -= len(extra)

    return selected[:limit]


def run_evaluation(questions: List[dict]):
    question_limit = get_question_limit()
    questions = sample_questions_by_category(questions, question_limit)
    results = []
    
    # Ensure current QA file is reset
    current_qa_path = os.path.join(os.path.dirname(__file__), "current_qa.txt")
    # Truncate/initialize the file
    with open(current_qa_path, "w") as _f:
        _f.write("")

    print(f"Running evaluation on {len(questions)} questions...")
    
    output_dir = os.path.join(os.path.dirname(__file__), "qa_responses")
    os.makedirs(output_dir, exist_ok=True)

    for i, q in enumerate(questions):
        category = q["category"]
        query = q["question"]
        expected_status = q["expected_status"]
        
        print(f"[{i+1}/{len(questions)}] [{category}] Query: {query[:50]}...")
        
        start_time = time.perf_counter()
        response: QueryResponse = process_query(query)
        latency = time.perf_counter() - start_time
        
        # Grading logic
        is_correct = False
        if expected_status == "Success" and response.status == "Success":
            is_correct = True # In a real scenario, we'd compare answer content
        elif expected_status == "Refused" and response.status == "Refused":
            is_correct = True
        elif expected_status == "False-Premise" and response.status == "False-Premise":
            is_correct = True
            
        item = {
            "category": category,
            "question": query,
            "expected_status": expected_status,
            "actual_status": response.status,
            "is_correct": int(is_correct),
            "latency": latency,
            "answer": response.answer
        }
        results.append(item)

        save_question_response(i + 1, category, query, expected_status, response)

        # Write current question/answer instantly for monitoring (flushed to disk)
        try:
            with open(current_qa_path, "w") as f:
                f.write(f"Question {i+1}/{len(questions)}\n")
                f.write(f"Category: {category}\n")
                f.write(f"Expected: {expected_status}\n")
                f.write(f"Status: {response.status}\n")
                f.write("Answer:\n")
                f.write(response.answer + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            # If writing the progress file fails, continue evaluation
            pass
    return results

def save_text_results(results, file_path="evaluation/results.txt"):
    with open(file_path, "w") as f:
        for i, item in enumerate(results, start=1):
            f.write(f"Question {i}: {item['question']}\n")
            f.write(f"Answer: {item['answer']}\n")
            f.write(f"Status: {item['actual_status']}\n")
            f.write("-" * 80 + "\n")


def calculate_metrics(results):
    df = pd.DataFrame(results)
    
    metrics = {}
    
    # Overall Accuracy
    metrics["Overall Accuracy"] = df["is_correct"].mean()
    
    # Per Category Accuracy
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        metrics[f"{cat} Accuracy"] = cat_df["is_correct"].mean()
        
    # Latency
    metrics["Mean Latency (s)"] = df["latency"].mean()
    metrics["P95 Latency (s)"] = df["latency"].quantile(0.95)
    
    # Refusal Precision/Recall (where expected_status was Refused)
    unanswerable = df[df["category"] == "Unanswerable"]
    if len(unanswerable) > 0:
        metrics["Refusal Recall"] = (unanswerable["actual_status"] == "Refused").mean()
        
    # False Premise Detection Rate
    fp = df[df["category"] == "False Premise"]
    if len(fp) > 0:
        metrics["False Premise Detection Rate"] = (fp["actual_status"] == "False-Premise").mean()

    return metrics

def main():
    questions = load_questions("evaluation/questions.json")
    results = run_evaluation(questions)
    
    # Save raw results
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save full question-answer text results
    save_text_results(results, file_path="evaluation/results.txt")
        
    # Print Metrics Table
    metrics = calculate_metrics(results)
    print("\n" + "="*40)
    print("METRICS REPORT")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<30}: {v:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
