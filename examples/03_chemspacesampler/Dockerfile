# Use an official Miniconda image as a parent image
#to install run: docker build -f Dockerfile -t chemspace ../.. in examples/03_chemspacesampler

FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app
ADD . /app

# Install any needed packages specified in requirements.txt
COPY examples/03_chemspacesampler/chemspace.yml /app
RUN conda env create -f chemspace.yml
# Make
# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "chemspace", "/bin/bash", "-c"]
# Make RUN commands use the new environment

# Make sure the environment is activated:
RUN echo "Make sure chemspace is activated:" && echo $CONDA_DEFAULT_ENV

# Activate the environment and install 'dscribe'
RUN echo "source activate chemspace" > ~/.bashrc
ENV PATH /opt/conda/envs/chemspace/bin:$PATH
RUN /bin/bash -c "source activate chemspace && conda install -c conda-forge dscribe"


# Define environment variable
ENV NAME chemspace


COPY mosaics /app
ENV PYTHONPATH "${PYTHONPATH}:/app/mosaics"
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
port = 8501\n\
" > /root/.streamlit/config.toml'
RUN echo 'machine-id' > /etc/machine-id
EXPOSE 8501
# command to run on container start
CMD [ "streamlit", "run", "examples/03_chemspacesampler/app.py"] 
#, "--server.fileWatcherType",  "none" ]
#docker run -p 8501:8051 chemspace