# Redrob Intelligent Candidate Ranking

## Overview

This project ranks candidates for the **Redrob Senior AI Engineer – Founding Team** role and generates the final submission CSV in the required format:

* `candidate_id`
* `rank`
* `score`
* `reasoning`

The system follows a **two-stage retrieval and ranking pipeline**.

* **Offline Precompute:** Generates semantic embeddings, candidate features, validation signals, and other reusable artifacts.
* **Online Ranking:** Loads the precomputed artifacts, retrieves the most relevant candidates, ranks them using multiple scoring signals, generates grounded reasoning, and produces the final submission CSV.

The ranking pipeline is designed to satisfy the competition constraints:

* CPU only
* No external API calls during ranking
* Runtime within the required limit
* Deterministic output

---

# Pipeline

```
Candidates JSONL
        │
        ▼
Offline Precompute
(Embeddings + Features + Validation)
        │
        ▼
Artifacts
        │
        ▼
Ranking Pipeline
        │
        ▼
Top 100 Candidates
        │
        ▼
Submission CSV
```

---

# Repository Structure

```
.
├── app.py                  # Streamlit demo
├── precompute.py           # Offline artifact generation
├── rank.py                 # Main ranking entry point
├── requirements.txt
├── README.md
│
├── config/
│   ├── config.yaml
│   └── lexicons.yaml
│
├── data/ (Create this folder and add candidates.jsonl inside it)
│   ├── candidates.jsonl 
├── src/
│   ├── retrieval.py
│   ├── core_fit.py
│   ├── behavioral.py
│   ├── fusion.py
│   ├── reasoning.py
│   ├── validator.py
│   └── ...
│
├── sample/
│   └── candidates_sample.jsonl
│
└── artifacts/
```

---

# Installation

Clone the repository.

```bash
git clone <repository-url>

cd <repository>
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1 — Generate Artifacts

Run this once for a new candidate dataset.

```bash
python precompute.py --candidates candidates.jsonl
```

---

## Step 2 — Generate Submission

```bash
python rank.py \
    --candidates candidates.jsonl \
    --out submission.csv
```

If artifacts are not available, the ranking pipeline automatically performs precomputation before generating the submission.

---

# Streamlit Demo

Cloud Demo Link: https://velora-f94q2n4nwe8uyekw9zswcr.streamlit.app/

OR

Launch the demo application.

```bash
streamlit run app.py
```

The demo allows you to:

* Upload a candidate dataset
* Execute the ranking pipeline
* Preview ranked candidates
* Download the generated submission CSV

---

# Configuration

Project configuration is stored under:

```
config/
```

* `config.yaml` – scoring parameters and pipeline configuration
* `lexicons.yaml` – keyword dictionaries and rule-based resources

---

# Output

The generated submission contains:

```
candidate_id
rank
score
reasoning
```

The output follows the Redrob submission specification.

---

# Compute Requirements

| Requirement                     | Status |
| ------------------------------- | ------ |
| CPU Only                        | ✅      |
| No Network Calls During Ranking | ✅      |
| Deterministic Ranking           | ✅      |
| Automatic CSV Validation        | ✅      |

---

# Notes

* Precomputation is performed only once per candidate dataset.
* The ranking stage uses only locally generated artifacts.
* Reasoning is generated directly from candidate profile information.
* The same input dataset always produces the same submission.
