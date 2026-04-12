"""
score.py — LLM-based student response scorer using OpenAI API.

Usage:
    python score.py --input <path_to_csv> [--output scored_output.csv] [--model gpt-4o]

The input CSV must contain these columns:
    Question, ResponseText.x, Model_Answer, Rubric, ResponseId, UNIV, Q_num, ground_truth
"""

import argparse
import sys
import pandas as pd
from openai import OpenAI

# ── Column name constants ──────────────────────────────────────────────────────
QUESTION_COL    = "Question"
STUDENT_ANS_COL = "ResponseText.x"
MODEL_ANS_COL   = "Model_Answer"
RUBRIC_COL      = "Rubric"

REQUIRED_COLS = [
    QUESTION_COL, STUDENT_ANS_COL, MODEL_ANS_COL, RUBRIC_COL,
    "ResponseId", "UNIV", "Q_num", "ground_truth",
]

PROMPT_TEMPLATE = """\
Context: You are a dedicated instructor for an introductory university-level \
statistics course. Students are given assessment questions during the course to \
help you diagnose how well they are learning, which will also help you plan any \
modifications to your instruction, to focus on statistics concepts the class \
seems to be struggling with. It is also important feedback for the students, \
since knowing what you know can be challenging. Given this goal, you use only \
three assessment categories: essentially correct, partially correct, incorrect. \
To make your assessment, you see the original question, the detailed assessment \
rubric you created at the time you developed the question, and the student's \
written answer.

Question:
{question}

Student Answer:
{student_answer}

Model Answer:
{model_answer}

Rubric:
{rubric}


Interpretation rules:
The rubric explicitly defines the required content for determining whether a \
response is incorrect, partially correct, or essentially correct.
The student does not need to use the exact wording from the rubric.
Minor grammatical errors, spelling mistakes, or informal phrasing should not \
reduce the score.
Correct statistical reasoning should receive credit even if the explanation is \
overly concise, or overly verbose.
Do not penalize differences in notation or terminology if the meaning is correct.

Output guidelines:
Output a 0 if the answer is incorrect, irrelevant, or does not satisfy the \
rubric criteria.
A score of 1 means the answer demonstrates partial understanding but is missing \
one or more essential elements required by the rubric.
A score of 2 means the answer correctly and completely addresses all essential \
elements of the rubric.

Important constraints:
Base your decision on the question, the rubric and the student's answer.
Output only a single integer, 0, 1, or 2.\
"""


def score_response(
    client: OpenAI,
    question: str,
    student_answer: str,
    model_answer: str,
    rubric: str,
    model: str,
) -> int | None:
    """Call the OpenAI API and return a score of 0, 1, or 2 (or None on error)."""
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        student_answer=student_answer,
        model_answer=model_answer,
        rubric=rubric,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    return int(raw) if raw in {"0", "1", "2"} else None


def main():
    parser = argparse.ArgumentParser(description="Score student responses with an LLM.")
    parser.add_argument("--input",  required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="LLM_q+a+m+r.csv", help="Path for output CSV (default: scored_output.csv)")
    parser.add_argument("--model",  default="gpt-5", help="OpenAI model to use (default: gpt-4o)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows.")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Input CSV is missing required columns: {missing}")

    # ── Set up client ──────────────────────────────────────────────────────────
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    total = len(df)
    scores = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        score = score_response(
            client,
            question=row[QUESTION_COL],
            student_answer=row[STUDENT_ANS_COL],
            model_answer=row[MODEL_ANS_COL],
            rubric=row[RUBRIC_COL],
            model=args.model,
        )
        scores.append(score)
        remaining = total - i
        print(f"\r✅ Row {i}/{total} scored: {score} | {remaining} remaining", end="", flush=True)

    print("\nScoring complete!")

    # ── Build output ───────────────────────────────────────────────────────────
    df["predicted_label"] = scores

    output_df = df[["ResponseId", "UNIV", "Q_num", "ground_truth", "predicted_label"]].copy()
    output_df = output_df.rename(columns={
        "ResponseId":    "response_id",
        "UNIV":          "university",
        "Q_num":         "question_type",
        "ground_truth":  "true_label",
    })

    print(f"\nPredicted label distribution:\n{output_df['predicted_label'].value_counts()}")
    print(f"\nTrue label distribution:\n{output_df['true_label'].value_counts()}")

    output_df.to_csv(args.output, index=False)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
