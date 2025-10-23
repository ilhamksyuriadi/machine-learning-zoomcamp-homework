FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /code

# Copy pyproject.toml and uv.lock for dependency installation
COPY pyproject.toml uv.lock ./

# Install uv (Python package manager)
RUN pip install uv

# Install dependencies from pyproject.toml using uv
RUN uv pip install --system -r pyproject.toml

# Copy the FastAPI application
COPY app.py .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]