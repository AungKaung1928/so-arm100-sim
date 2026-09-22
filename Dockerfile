# Reproducible, CPU-only. Runs the test suite by default.
#   docker build -t so-arm100-sim .
#   docker run --rm so-arm100-sim
# Rendering tests skip inside the container: it has no GL context. Install
# libegl/libosmesa and set MUJOCO_GL if you need images from Docker.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY so_arm100_sim ./so_arm100_sim
COPY tests ./tests
COPY scripts ./scripts
COPY verify.sh ./

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "tests"]
