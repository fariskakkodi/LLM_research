# LLM Output Statistics — Student Response Scorer

Automatically scores student short-answer responses for an introductory statistics course using the OpenAI API. Each response receives a score of **0** (incorrect), **1** (partially correct), or **2** (essentially correct) based on the question, a model answer, and a rubric.

---

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd llm_output_statistics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Usage

```bash
python score.py --input <your_data.csv> [--output scored_output.csv] [--model gpt-5]
```

### Arguments

| Argument   | Required | Default             | Description                          |
|------------|----------|---------------------|--------------------------------------|
| `--input`  | ✅       | —                   | Path to your input CSV file          |
| `--output` | ❌       | `scored_output.csv` | Where to save the results            |
| `--model`  | ❌       | `gpt-4o`            | OpenAI model to use for scoring      |

### Example
```bash
python score.py --input data/test_new_data_final.csv --output results/scored.csv --model gpt-4o
```

---

## Input CSV Format

Your CSV must contain these columns:

| Column           | Description                              |
|------------------|------------------------------------------|
| `Question`       | The assessment question text             |
| `ResponseText.x` | The student's written answer             |
| `Model_Answer`   | The instructor's model answer            |
| `Rubric`         | The scoring rubric                       |
| `ResponseId`     | Unique identifier for the response       |
| `UNIV`           | University identifier                    |
| `Q_num`          | Question number / type                   |
| `ground_truth`   | Human-assigned ground truth label (0/1/2)|

---

## Output CSV Format

| Column           | Description                       |
|------------------|-----------------------------------|
| `response_id`    | Original `ResponseId`             |
| `university`     | Original `UNIV`                   |
| `question_type`  | Original `Q_num`                  |
| `true_label`     | Original `ground_truth`           |
| `predicted_label`| LLM-assigned score (0, 1, or 2)   |

---

## Scoring Rubric

The LLM is prompted to act as a statistics instructor and score each response as:

- **0** — Incorrect or irrelevant
- **1** — Partially correct (missing one or more essential elements)
- **2** — Essentially correct (addresses all essential rubric elements)

Minor spelling/grammar issues, informal phrasing, or notation differences are not penalized.

---

## Notes

- The script prints live progress as rows are scored.
- Rows where the model returns an unexpected value are saved as `None` in `predicted_label`.
- Default model is `gpt-4o`. You can switch to `gpt-4-turbo`, `gpt-3.5-turbo`, etc. via `--model`.
