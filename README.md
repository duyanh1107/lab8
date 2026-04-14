## Answer Pipeline Submission

This folder contains the current query-to-grounded-answer pipeline extracted from `adaptive-learning-ai`.

### Main entrypoint

- `debug_answer.py`

Run from this folder:

```powershell
python debug_answer.py
```

### Environment

Create a local `.env` file from `.env.example` and set:

```text
OPENAI_API_KEY=...
```

### Included pipeline stages

1. Query transform selection
2. Query transformation
   - expansion
   - decomposition
   - step-back
   - HyDE
3. Hybrid retrieval
   - dense vector retrieval
   - BM25 sparse retrieval
   - weighted RRF fusion
4. Reranking
5. Grounded context building
6. Grounded answer prompt building
7. Final answer generation

### Main folders

- `llm/`
  Prompting, query transforms, reranker, and answer-prompt builder.
- `rag/`
  Chunking, document processing, retrieval, vector store, grounding, and service orchestration.
- `services/`
  Course/module lookup used by the debug entrypoint.
- `core/`
  Shared data models.
- `data/`
  Course config, skills, TOC, and the source PDF used by the current math pipeline.

### Notes

- The reranker uses a cross-encoder if the local dependencies are installed.
- If `sentence-transformers`, `transformers`, or `torch` are missing, reranking falls back to lexical overlap.
- This folder is a submission snapshot of the current answer-generation pipeline, not the whole app.
