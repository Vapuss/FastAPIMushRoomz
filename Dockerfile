# 1. Imagine de bază
FROM python:3.10

# 2. Setăm directorul de lucru
WORKDIR /app

# 3. Copiem backendul în container
COPY ./backend /app

# 4. Instalăm librăriile
RUN pip install --no-cache-dir -r requirements.txt

# 5. Pornim serverul FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
