# # Base image
# FROM python:3.10.7

# # Copy files
# COPY . /FBHP_app

# # Set working directory
# WORKDIR /FBHP_app

# # Install dependencies
# RUN pip install -r requirements.txt

# # Expose port
# EXPOSE 8080

# # # Start Streamlit app
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

# Base image
FROM python:3.10.7

# Copy files
COPY . /FBHP_app

# Set working directory
WORKDIR /FBHP_app

# Install dependencies
RUN pip install -r requirements.txt

# Start Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]



 