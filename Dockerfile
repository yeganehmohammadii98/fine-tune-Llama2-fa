FROM runpod/pytorch:2.1.1-py3.10-cuda11.8.0

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your training script and data
COPY train.py .
COPY combined_texts.txt .
