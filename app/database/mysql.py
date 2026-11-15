import os
from collections.abc import Iterator
from contextlib import contextmanager

import pymysql
from dotenv import load_dotenv
from pymysql.connections import Connection

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_connection() -> Connection:
    """Open a MySQL connection using MYSQL_* environment variables."""
    return pymysql.connect(
        host=_require_env("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=_require_env("MYSQL_USER"),
        password=_require_env("MYSQL_PASSWORD"),
        database=_require_env("MYSQL_DB"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


@contextmanager
def mysql_connection() -> Iterator[Connection]:
    """Yield a connection, commit on success, roll back on error, and always close."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
