FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install langchain-community
COPY . .

CMD ["python", "agent_test.py"]