# # Base image 
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /probhp

# Copy the dependencies file to the working directory
COPY . .
# Dont save the downloaded packages locally
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
