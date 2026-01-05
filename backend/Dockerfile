# Backend service container
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Reduce thread usage and avoid TLS allocation failures in constrained containers.
    ARROW_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OMP_THREAD_LIMIT=1 \
    OMP_DYNAMIC=FALSE \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MKL_THREADING_LAYER=SEQUENTIAL \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1 \
    PANDAS_USE_NUMEXPR=0 \
    GLIBC_TUNABLES=glibc.rtld.dynamic_tls_surplus=256

WORKDIR /app

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1

RUN pip install --no-cache-dir --upgrade pip

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./

EXPOSE 8105

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8105"]
