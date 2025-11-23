FROM python:3.12-slim

WORKDIR /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install asyncio and streamlit first
#RUN pip install --no-cache-dir asyncio==3.4.3 && \
#RUN  pip install --no-cache-dir streamlit==1.31.0
RUN  pip install --no-cache-dir streamlit==1.47.1

# Install packages in specific order to resolve dependencies
RUN pip install --no-cache-dir mcp==1.9.4 && \
    #pip install --no-cache-dir fastmcp==2.7.1 && \
    pip install --no-cache-dir fastmcp==2.7.0 && \
    pip install --no-cache-dir reward-kit==0.3.4 && \
    pip install --no-cache-dir langchain-mcp-adapters==0.1.7

# Copy requirements.txt
COPY requirements.txt .

# Create a script to filter requirements
RUN echo 'import sys\n\
with open("requirements.txt") as f:\n\
    reqs = f.readlines()\n\
filtered = [r for r in reqs if not any(x in r.lower() for x in \
    ["streamlit", "asyncio", "mcp", "fastmcp", "reward-kit", "langchain-mcp-adapters"])]\n\
with open("filtered_requirements.txt", "w") as f:\n\
    f.writelines(filtered)' > filter_requirements.py

# Run the filter script and install filtered requirements
RUN python filter_requirements.py && \
    pip install --no-cache-dir -r filtered_requirements.txt

RUN  pip install --no-cache-dir streamlit-authenticator==0.1.5
RUN  pip install --no-cache-dir statsmodels==0.14.5
RUN  pip install --no-cache-dir streamlit-js-eval==0.1.5
RUN  pip install --no-cache-dir deepagents==0.0.3
RUN  pip install --no-cache-dir streamlit_autorefresh==1.0.1

# Copy all files
COPY . .

# Set PYTHONPATH to include /app and the virtual environment's site-packages
ENV PYTHONPATH=/app:/opt/venv/lib/python3.12/site-packages



# Set working directory and Python path
#ENV PYTHONPATH=/opt/venv/lib/python3.12/site-packages:/app
#WORKDIR /app/

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run Streamlit
#CMD ["streamlit", "run", "mcp_client/stock_agent_combined_ui_working_copy.py", "--server.address=0.0.0.0", "--server.port=8501"]

#CMD ["/opt/venv/bin/python", "-m", "streamlit", "run", "mcp_client/stock_agent_combined_ui_working_copy.py", "--server.address=0.0.0.0", "--server.port=8501"]
#for entry as main.py
CMD ["/opt/venv/bin/python", "-m", "streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
