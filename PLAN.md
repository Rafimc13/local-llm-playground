# Implementation roadmap

## Phase 1 — common architecture

Complete: installable uv project, model registry, typed schemas, backend protocol,
FastAPI/CLI entry points, runtime profiles, CPU-only tests, and CI.

## Phase 2 — Ollama experiments

User-run work: install Ollama, pull each Q4_K_M checkpoint, validate chat and
streaming, and capture placement with `ollama ps` plus GPU telemetry.

## Phase 3 — vLLM and SGLang experiments

User-run work: launch one pinned container at a time, establish 0.6B and 1.7B
low-memory settings, and record any driver/kernel incompatibility.

## Phase 4 — optimization matrix

User-run work: compare FP16, Q4_K_M, and AWQ where supported; then vary offload,
context, batching, chunked prefill, cache behavior, and CUDA graphs.

## Phase 5 — results

Extend the checked-in bilingual prompt dataset, export JSON/CSV benchmark records,
and document the exact backend version, checkpoint revision, and GPU environment.
