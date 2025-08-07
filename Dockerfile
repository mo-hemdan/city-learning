# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Optional: If you want to install your package
RUN pip install -e .

# Default command
CMD ["python", "main.py"]
