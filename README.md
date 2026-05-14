# Hybrid Retrieval for AI FAQ Optimization

This repository contains experiments for improving FAQ retrieval using embedding-based semantic search and BM25-based lexical retrieval.

The project focuses on question retrieval for a cleaned AI FAQ dataset. It compares semantic retrieval based on multilingual E5 embeddings with lexical retrieval based on BM25, and explores whether combining both methods can improve retrieval coverage.

## Project Overview

FAQ retrieval often faces two types of matching problems:

1. **Semantic mismatch**:  
   A user may ask a question using different wording from the original reference question.

2. **Keyword mismatch**:  
   Some relevant answers may contain important terms that are not obvious from the question title alone.

To address these issues, this project evaluates a hybrid retrieval pipeline that combines:

- Embedding-based semantic retrieval
- BM25 keyword-based retrieval
- Candidate comparison between the two methods

## Dataset

The reference dataset contains cleaned question-answer pairs from the FAQ collection.

The main dataset file is:

```text
data/reference_qa_clean.csv
