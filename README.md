# RAG for Philanthropic Campaign Analysis

[![Open Notebook 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIforPhilanthropy/RAG/blob/main/campaign_embeddings_bert_mini.ipynb)
[![Open Notebook 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIforPhilanthropy/RAG/blob/main/rag_llm_benchmark_judge.ipynb)

## Project Overview

This repository contains an end-to-end educational workflow for Retrieval-Augmented Generation (RAG) applied to philanthropic campaign data.

The workflow is designed for teaching and experimentation. It covers:
- Building campaign-level embeddings from structured campaign records.
- Retrieving relevant campaign evidence for user questions.
- Generating grounded answers with selected LLMs.
- Benchmarking multiple answer models on the same question set.
- Evaluating answers with an LLM-as-judge rubric.
- Visualizing model performance by metric, by question, and overall.
- Running statistical tests to assess whether observed differences are significant.

The notebooks are especially suited for coursework and lab sessions where students need both conceptual explanations and executable code.

## Repository Contents

### Core Input Data
- `06_method_b_pagination_and_records.json`
  - Source dataset containing campaign records.

### Notebook 1: Embeddings + RAG Construction
- `campaign_embeddings_bert_mini.ipynb`
  - Parses and structures campaign records.
  - Detects campaign instances.
  - Generates embeddings with `prajjwal1/bert-mini`.
  - Saves reusable retrieval artifacts.
  - Includes interactive projection/visual exploration.

Open in Colab:
- https://colab.research.google.com/github/AIforPhilanthropy/RAG/blob/main/campaign_embeddings_bert_mini.ipynb

Why this notebook matters (rationale):
- It creates the full retrieval foundation used by the benchmark notebook.
- It standardizes campaign text, metadata, and instance detection so downstream evaluation is consistent.
- It produces reusable artifacts (CSV, NPY, JSONL, Parquet) that can be consumed by many RAG systems.
- It helps students understand the difference between building a retriever and benchmarking generators.
- It allows visual sanity checks of embedding quality before any model comparison.

### Notebook 2: Multi-LLM Benchmark + LLM-as-Judge
- `rag_llm_benchmark_judge.ipynb`
  - Loads prepared artifacts.
  - Uses a single source-of-truth model selection cell.
  - Runs RAG answer generation for selected models.
  - Uses a dedicated judge model for rubric-based scoring.
  - Exports answer logs, judged logs, winners, leaderboard, and significance results.
  - Includes comparative visualizations and statistical significance analysis.

Open in Colab:
- https://colab.research.google.com/github/AIforPhilanthropy/RAG/blob/main/rag_llm_benchmark_judge.ipynb

Why this notebook comes second:
- It assumes Notebook 1 outputs already exist.
- It focuses on controlled comparison of answer models, not retriever construction.
- It adds evaluation, interpretation, and significance testing on top of the retrieval pipeline.

### Generated Artifacts
- `campaign_instances.csv`
- `campaign_embeddings_bert_mini.npy`
- `campaign_embeddings_rag.jsonl`
- `campaign_embeddings_rag.parquet`
- `rag_benchmark_answers.jsonl`
- `rag_benchmark_answers.csv`
- `rag_benchmark_judged.jsonl`
- `rag_benchmark_judged.csv`
- `rag_benchmark_winners_per_question.csv`
- `rag_benchmark_leaderboard.csv`
- `rag_benchmark_significance_tests.csv`

## Recommended Execution Order

1. Run Notebook 1 (`campaign_embeddings_bert_mini.ipynb`) first.
2. Verify generation of retrieval artifacts (especially `campaign_instances.csv` and `campaign_embeddings_bert_mini.npy`).
3. Run Notebook 2 (`rag_llm_benchmark_judge.ipynb`) second.

## Methodological Flow

## 1) Data Ingestion and Structuring
The pipeline ingests campaign records from JSON and builds structured campaign text fields. It preserves key metadata like title and URL so generated answers can cite evidence.

## 2) Embedding Space Construction
Campaign texts are embedded with a compact encoder (`bert-mini`) so that both retrieval quality and compute cost remain reasonable for classroom use.

## 3) Retrieval
At query time, user questions are embedded in the same vector space and matched to campaign vectors via cosine similarity. Top-k records are used as the evidence context.

## 4) RAG Answering (Notebook 2)
Each answer model receives:
- The same question.
- The same retrieved context.

This ensures fair comparative evaluation by controlling retrieval variance.

## 5) LLM-as-Judge Evaluation (Notebook 2)
A separate judge model scores each answer on:
- factuality
- conciseness
- groundedness
- completeness
- uncertainty_handling

A weighted overall score is computed and stored with reasoning text when available.

## 6) Comparative Analytics (Notebook 2)
The benchmark notebook provides:
- Per-dimension model comparison by question.
- Overall score comparison by question.
- Heatmap view (model x question).
- Per-question winners.
- Aggregate leaderboard.

## 7) Statistical Significance (Notebook 2)
The notebook also computes pairwise significance testing between models:
- two-sided sign test
- exact paired sign-flip test
- Wilcoxon signed-rank (if available)
- effect size (`Cohen's d_z`)

This supports stronger interpretation beyond raw mean differences.

## Model Configuration (Current)
Model selection is centralized in Notebook 2 under Section 3.1.

Current setup:
- Answer model A: `gemma-4-31b-it`
- Answer model B: `gemma-4-26b-a4b-it`
- Judge model: `gemini-3.1-flash-lite`

All downstream generation calls are constrained to models defined in that single cell.

## Run Instructions

## Local (Jupyter)
1. Open `campaign_embeddings_bert_mini.ipynb` and run all cells.
2. Confirm generation of campaign artifacts (`campaign_instances.csv`, embeddings `.npy`, and optional exports).
3. Open `rag_llm_benchmark_judge.ipynb`.
4. Set API keys through `.env` and/or Colab secrets compatible logic.
5. Run cells in order.
6. Review exported benchmark and significance outputs.

## Google Colab
Use these notebook links in sequence:
- Notebook 1: https://colab.research.google.com/github/AIforPhilanthropy/RAG/blob/main/campaign_embeddings_bert_mini.ipynb
- Notebook 2: https://colab.research.google.com/github/AIforPhilanthropy/RAG/blob/main/rag_llm_benchmark_judge.ipynb

If running in Colab:
- Ensure required packages are installed from the optional install cell.
- Provide API keys via Colab secrets or environment variables.
- Mount any required data files if not already in the repository path.

## Required Python Packages
Typical dependencies used across notebooks:
- pandas
- numpy
- torch
- transformers
- tqdm
- python-dotenv
- requests
- openai
- anthropic
- google-generativeai
- osfclient
- plotly
- scipy

## Evaluation Interpretation Guidance
- High score does not imply universal superiority.
- Scores can be sensitive to judge behavior and question set size.
- For small question sets, significance tests have low power.
- Pair p-values with effect sizes and qualitative review.

## Reproducibility Notes
- Keep model selection fixed before running benchmarks.
- Keep question set unchanged when comparing runs.
- Use exported CSV/JSONL outputs for audit trails.
- Record package versions and API model versions for longitudinal comparisons.

## Limitations
- LLM judges are useful but imperfect evaluators.
- API availability/rate limits can affect run stability.
- Very small benchmark sets reduce inferential strength.

## Suggested Extensions
- Add more benchmark questions per difficulty tier.
- Add human adjudication alongside LLM judging.
- Add bootstrap confidence intervals for mean differences.
- Track longitudinal drift with repeated benchmark runs.
- Add cost and latency dashboards per model.

## Ethics and Responsible Use
This project is intended for research and teaching. Generated outputs should not be treated as definitive factual truth without verification against primary sources.

When used for nonprofit decision support:
- keep humans in the loop,
- verify critical claims,
- preserve transparent source citations,
- document model and prompt configurations.
