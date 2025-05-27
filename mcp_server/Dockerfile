FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for uv installation
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy pyproject.toml and uv.lock for dependency installation
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY graphiti_mcp_server.py .
COPY memory_enhanced_server.py .
COPY memory_entity_types.py .
COPY server_wrapper.py .

EXPOSE 8000

# Set environment variables for better error handling
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/sse --max-time 5 || exit 1

# Command to run the application with proper signal handling via wrapper
CMD ["uv", "run", "server_wrapper.py"]
