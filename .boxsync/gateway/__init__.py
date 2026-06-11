"""EXACT 2026 submission gateway.

A single FastAPI app exposing the competition `/predict` endpoint. It routes each
query by `type`:
    type1  -> logic pipeline  (cascade prompts + solver, via the shared vLLM model)
    type2  -> physics pipeline (exact_fama ExactFamaPipeline, via the shared vLLM model)

and proxies `/v1/models` to the underlying vLLM server so a single public URL
covers both the prediction endpoint and the model-verification endpoint.
"""
