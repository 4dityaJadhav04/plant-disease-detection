FROM python:3.10-slim

WORKDIR /app

# Copy only what's needed
COPY app/ ./app/

# Set up Streamlit config
RUN mkdir -p /root/.streamlit
COPY app/config.toml /root/.streamlit/config.toml
COPY app/credentials.toml /root/.streamlit/credentials.toml

# Install dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app/main.py"]
