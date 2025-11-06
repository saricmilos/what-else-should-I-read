# Step 1: Base image
FROM python:3.11

# Step 2: Set working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy all project files into the container
COPY . /app

# Step 5: Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Expose the port FastAPI will run on
EXPOSE 8000

# Step 7: Command to run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
