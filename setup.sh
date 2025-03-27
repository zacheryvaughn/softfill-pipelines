python -m venv .venv && \
source .venv/bin/activate && \
pip install --upgrade pip && \
pip cache purge && \
pip install torch diffusers transformers && \
pip freeze > requirements.txt