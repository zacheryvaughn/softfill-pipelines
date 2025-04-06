# Windows python
# Mac/Linux python3
python -m venv .venv && \
source .venv/bin/activate && \
pip install --upgrade pip && \
pip cache purge && \
pip install torch torchvision diffusers transformers noise opencv-python && \
pip freeze > requirements.txt