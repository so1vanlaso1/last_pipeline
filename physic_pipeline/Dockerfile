FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    LLM_BACKEND=none \
    EXACT_FAMA_CONFIG=configs/eval_solver_only.yaml \
    EXACT_USE_PROVIDED_FOL=0

COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./README.md

RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "exact_fama.api:app", "--host", "0.0.0.0", "--port", "8000"]
