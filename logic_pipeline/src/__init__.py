"""NEWpipeline — weighted soft-vote NL-QA pipeline.

Up to three stages each answer EVERY question, and the answers are combined by a
weighted vote (4B model → weight 1.0, 8B model → weight 1.5):

    stage "4b"        Qwen3.5-4B + Gemma-E2B
    stage "gemma8b"   Gemma-E4B
    stage "liquid8b"  LFM2.5-8B-A1B

Stages run one at a time — each stage's model(s) load, vote on every record, then
are UNLOADED before the next stage loads — so at most {two 4B models} OR {one 8B
model} are resident at a time and peak VRAM stays bounded. Pick the line-up with
run_cascade.py's --stages flag.
"""
