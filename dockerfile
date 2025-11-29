FROM python:3.12-slim

WORKDIR /app

# Evita errores por locale
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependencias del sistema para PyMuPDF
RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx poppler-utils && \
    apt-get clean

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
