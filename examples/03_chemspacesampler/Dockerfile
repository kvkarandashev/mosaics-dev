# Pull the base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

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
port = $PORT\n\
" > /root/.streamlit/config.toml'

# command to run on container start
CMD [ "streamlit", "run", "app.py" ]
