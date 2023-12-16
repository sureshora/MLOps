# Makefile

install:
    pip install -r requirements.txt

test:
    python -m pytest tests/

train:
    python scripts/train_model.py

deploy:
    python scripts/deploy_model.py
