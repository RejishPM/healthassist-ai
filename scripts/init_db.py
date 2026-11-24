"""Explicitly initialize/upgrade the configured demo database; never run on import."""

import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg


def main():
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is required")
    schema = Path(__file__).resolve().parents[1] / "app/db/schema.sql"
    with psycopg.connect(url) as connection:
        connection.execute(schema.read_text())
    print("Database schema initialized/upgraded")


if __name__ == "__main__":
    main()
