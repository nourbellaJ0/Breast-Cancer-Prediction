FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

COPY . .

EXPOSE 8000 8501

COPY start.sh /start.sh
RUN chmod +x /start.sh


CMD ["/start.sh"]
