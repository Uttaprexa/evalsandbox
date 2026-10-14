FROM python:3.11-slim
# Only stdlib is available to candidates. Add numpy etc. here if your benchmark needs it.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY harness.py /opt/harness/harness.py
RUN chmod 555 /opt/harness && chmod 444 /opt/harness/harness.py
# Starts as root ONLY so the harness can spawn the candidate as uid 65534 and then drop
# itself to uid 1000. The container gets no capabilities except SETUID/SETGID.
WORKDIR /
ENTRYPOINT ["python", "-I", "/opt/harness/harness.py"]
