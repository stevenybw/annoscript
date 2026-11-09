FROM python:3.12-slim-bookworm
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY annoscript ./annoscript
COPY annoscript_examples ./annoscript_examples
RUN pip install --no-cache-dir '.[server]' && \
    useradd --uid 10001 --create-home anno && mkdir /data && chown anno:anno /data && rm -rf /src
WORKDIR /app
USER anno
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python", "-m", "annoscript.server", "gateway", "--host", "0.0.0.0", "--port", "8000"]
