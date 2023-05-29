# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install necessary system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository and install the mosaics library
RUN git clone https://github.com/kvkarandashev/mosaics.git
ENV PYTHONPATH "${PYTHONPATH}:/mosaics"

# Add the current directory contents into the container
ADD . /mosaics
ADD ./examples /app/examples
ADD ./misc_scripts /app/misc_scripts

# Install Python dependencies
RUN pip install --no-cache-dir -r /mosaics/requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit_app.py when the container launches
CMD streamlit run /app/examples/03_chemspacesampler/app.py