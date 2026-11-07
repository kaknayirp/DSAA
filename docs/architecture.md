# DSAA Architecture

## Project
Dynamic Smart Attention-based Allocation

## Objective
Reduce LLM KV Cache memory consumption
while preserving answer quality.

## Modules

1. LLM Engine
   - Model loading
   - KV Cache extraction
   - Cache management

2. DSAA Algorithm
   - Attention scoring
   - Semantic scoring
   - Recency scoring
   - Memory-aware retention

3. Frontend and Evaluation
   - React dashboard
   - Benchmarking
   - Memory measurement
   - Accuracy evaluation

4. Backend
   - FastAPI
   - Module integration

## Initial Model
Qwen2.5-1.5B-Instruct

## Framework
Python, PyTorch, Hugging Face Transformers