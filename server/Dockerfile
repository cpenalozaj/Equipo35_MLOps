# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY logistic_regression.pkl /app/
COPY requirements.txt /app/
# Install dependencies
RUN pip install -r requirements.txt

COPY . /app/
# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]