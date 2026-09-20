FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry && poetry config virtualenvs.create false && poetry install --only main

COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", \
               "--host", "0.0.0.0", \
                 "--port", "8000", \
                 "--reload"]