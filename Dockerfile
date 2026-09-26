# Auriva trajectory + phoneme microservice (Flask, port 5001).
# Python 3.13 matches the local .venv the trajectory pickle was tested with.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY . .

# Fail the build if the model cannot be unpickled — app.py only logs that at
# startup and then serves 503s.
RUN python -c "import joblib; a = joblib.load('models/trajectory_model_calibrated.pkl'); print('trajectory model ok:', sorted(a))"

RUN useradd --uid 1000 --no-create-home app
USER app
EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=4)"

# app.run() in app.py binds 127.0.0.1 only — unreachable from other containers.
# gunicorn binds all interfaces; 2 workers fit the 2-vCPU VM.
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5001", "--timeout", "60", "--access-logfile", "-", "app:app"]
