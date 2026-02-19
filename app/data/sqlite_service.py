import sqlite3
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    declared_type: str


class SQLiteService:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).expanduser())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def list_tables(self) -> List[str]:
        sql = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """
        with self._connect() as connection:
            rows = connection.execute(sql).fetchall()
        return [row["name"] for row in rows]

    def table_exists(self, table_name: str) -> bool:
        sql = """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            LIMIT 1
        """
        with self._connect() as connection:
            row = connection.execute(sql, (table_name,)).fetchone()
        return row is not None

    def list_columns(self, table_name: str) -> List[ColumnInfo]:
        sql = f"PRAGMA table_info({quote_identifier(table_name)})"
        with self._connect() as connection:
            rows = connection.execute(sql).fetchall()
        return [
            ColumnInfo(name=row["name"], declared_type=(row["type"] or ""))
            for row in rows
        ]

    def fetch_preview(self, table_name: str, limit: int = 50) -> Tuple[List[str], List[sqlite3.Row]]:
        sql = f"SELECT * FROM {quote_identifier(table_name)} LIMIT ?"
        with self._connect() as connection:
            cursor = connection.execute(sql, (limit,))
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description or []]
        return column_names, rows

    def fetch_time_bounds(
        self,
        table_name: str,
        time_column: str = "time@timestamp",
    ) -> Optional[Tuple[float, float]]:
        q_table = quote_identifier(table_name)
        q_time = quote_identifier(time_column)
        sql = (
            f"SELECT MIN({q_time}) AS min_time, MAX({q_time}) AS max_time "
            f"FROM {q_table} "
            f"WHERE {q_time} IS NOT NULL"
        )
        with self._connect() as connection:
            row = connection.execute(sql).fetchone()
        if not row:
            return None
        min_time = row["min_time"]
        max_time = row["max_time"]
        if min_time is None or max_time is None:
            return None
        return float(min_time), float(max_time)

    def fetch_series_rows(
        self,
        table_name: str,
        y_column: str,
        limit: int,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> Iterable[sqlite3.Row]:
        q_table = quote_identifier(table_name)
        q_time = quote_identifier("time@timestamp")
        q_y = quote_identifier(y_column)
        conditions = [f"{q_time} IS NOT NULL", f"{q_y} IS NOT NULL"]
        params: List[object] = []

        if start_ts is not None:
            conditions.append(f"{q_time} >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            conditions.append(f"{q_time} <= ?")
            params.append(float(end_ts))

        where_clause = " AND ".join(conditions)
        sql = (
            f"SELECT {q_time} AS time_value, {q_y} AS y_value "
            f"FROM {q_table} "
            f"WHERE {where_clause} "
            f"ORDER BY {q_time} "
            f"LIMIT ?"
        )
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return rows

    def fetch_multi_series_rows(
        self,
        table_name: str,
        y_columns: List[str],
        limit: int,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> Iterable[sqlite3.Row]:
        if not y_columns:
            return []

        q_table = quote_identifier(table_name)
        q_time = quote_identifier("time@timestamp")
        select_parts = [f"{q_time} AS time_value"]
        for column in y_columns:
            q_column = quote_identifier(column)
            select_parts.append(f"{q_column} AS {quote_identifier(column)}")

        conditions = [f"{q_time} IS NOT NULL"]
        params: List[object] = []
        if start_ts is not None:
            conditions.append(f"{q_time} >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            conditions.append(f"{q_time} <= ?")
            params.append(float(end_ts))

        non_null_any = " OR ".join(f"{quote_identifier(col)} IS NOT NULL" for col in y_columns)
        conditions.append(f"({non_null_any})")

        where_clause = " AND ".join(conditions)
        select_clause = ", ".join(select_parts)
        sql = (
            f"SELECT {select_clause} "
            f"FROM {q_table} "
            f"WHERE {where_clause} "
            f"ORDER BY {q_time} "
            f"LIMIT ?"
        )
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return rows

    def ensure_time_index(
        self,
        table_name: str,
        time_column: str = "time@timestamp",
    ) -> str:
        raw_index_name = f"idx_{table_name}_{time_column}_ts"
        safe_index_name = re.sub(r"[^0-9A-Za-z_]+", "_", raw_index_name).strip("_")
        if not safe_index_name:
            safe_index_name = "idx_time_timestamp"

        q_index = quote_identifier(safe_index_name)
        q_table = quote_identifier(table_name)
        q_time = quote_identifier(time_column)
        sql = f"CREATE INDEX IF NOT EXISTS {q_index} ON {q_table} ({q_time})"

        with self._connect() as connection:
            connection.execute(sql)
            connection.commit()
        return safe_index_name
