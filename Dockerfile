# 1) Base slim
FROM python:3.12-slim

# 2) Security: no root user
RUN useradd -m -u 10001 appuser

# 3) Workdir
WORKDIR /app

# 4) Install deps (copy first for better cache)
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 5) Copy source
COPY app /app/app
COPY data /app/data

# 6) Env
ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# 7) Switch to non-root
USER appuser

EXPOSE 5000

# 8) Run
CMD ["python", "-m", "flask", "run"]
