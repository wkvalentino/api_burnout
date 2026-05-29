FROM python:3.11-slim

WORKDIR /app

# Instal dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua kode dan aset model
COPY . .

# Jalankan Uvicorn dengan port dinamis dari Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]