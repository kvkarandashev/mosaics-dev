# Pull the base image
FROM continuumio/miniconda3:latest
# Update conda
RUN conda update -n base -c defaults conda

RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenblas-base



# Install xyz2mol
RUN conda install -c conda-forge xyz2mol
# Set the working directory
WORKDIR /app
ADD . /app

# Copy the dependencies file to the working directory
COPY requirements.txt .
RUN pip install protobuf==3.20.0
RUN pip install rdkit-pypi

COPY mosaics /app
ENV PYTHONPATH "${PYTHONPATH}:/app/mosaics"
# Install any dependencies
RUN conda install -c conda-forge numpy
RUN pip install --no-cache-dir -r requirements.txt
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV STREAMLIT_TELEMETRY_OPT_OUT=true
# Copy the content of the local src directory to the working directory
COPY . .

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = 80\n\
" > /root/.streamlit/config.toml'
RUN echo 'machine-id' > /etc/machine-id
# command to run on container start
CMD [ "streamlit", "run", "examples/03_chemspacesampler/app.py", "--server.fileWatcherType",  "none" ]

