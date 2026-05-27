FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY data ./data
COPY tests ./tests

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
