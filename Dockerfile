# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for scikit-learn
RUN pip install --no-cache-dir flask gunicorn

# Copy all project files to container
COPY . .

# Create directory for templates
RUN mkdir -p templates

# Expose port 5000 for Flask app
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
