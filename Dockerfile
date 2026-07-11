# Use the EXACT same Python version you used for training
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to cache dependencies
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary folders into the container
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]