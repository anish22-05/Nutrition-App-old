FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

# dependencies first, so code edits don't invalidate this (slow) layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY front.py .
COPY .streamlit ./.streamlit

# secrets come from the environment (see .env.example), never from the image
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
