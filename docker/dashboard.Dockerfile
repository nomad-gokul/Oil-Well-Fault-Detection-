# syntax=docker/dockerfile:1
# Dashboard image: Streamlit + the thin API client only.
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements-dashboard.txt .
RUN pip install -r requirements-dashboard.txt
COPY src/owfd/__init__.py ./owfd/__init__.py
COPY src/owfd/dashboard ./owfd/dashboard
ENV PYTHONPATH=/app OWFD_API_URL=http://api:8000
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request as u; sys.exit(u.urlopen('http://localhost:8501/_stcore/health').status != 200)"
CMD ["sh", "-c", "streamlit run owfd/dashboard/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
