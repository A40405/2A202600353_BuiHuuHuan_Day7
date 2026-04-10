FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV GITHUB_TOKEN=your_token_here

CMD ["python", "src/run_agent.py"]