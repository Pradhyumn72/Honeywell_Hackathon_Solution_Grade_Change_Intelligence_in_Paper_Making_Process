# """
# database.py
# -----------
# Lightweight SQLite wrapper for the Intelligent Grade Change System.
# Handles creation of the operator_feedback.db and all CRUD operations
# on the feedback_logs table.
# """

# import sqlite3
# import os
# import logging

# # --- Configuration ---
# DB_PATH = "operator_feedback.db"

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)


# def get_connection() -> sqlite3.Connection:
#     """
#     Establishes and returns a connection to the SQLite database.
#     Creates the database file if it does not already exist.
#     """
#     conn = sqlite3.connect(DB_PATH, check_same_thread=False)
#     conn.row_factory = sqlite3.Row  # Enables column access by name
#     return conn


# def initialize_database():
#     """
#     Creates the feedback_logs table if it does not already exist.
#     Safe to call multiple times (idempotent).
#     """
#     create_table_sql = """
#     CREATE TABLE IF NOT EXISTS feedback_logs (
#         id                      INTEGER PRIMARY KEY AUTOINCREMENT,
#         timestamp               TEXT    NOT NULL,
#         current_basis_weight    REAL    NOT NULL,
#         predicted_basis_weight  REAL    NOT NULL,
#         suggested_stock_flow    REAL    NOT NULL,
#         suggested_steam_pressure REAL   NOT NULL,
#         operator_decision       TEXT    NOT NULL CHECK(operator_decision IN ('Accepted', 'Rejected'))
#     );
#     """
#     try:
#         with get_connection() as conn:
#             conn.execute(create_table_sql)
#             conn.commit()
#         logger.info("Database initialized. Table 'feedback_logs' is ready.")
#     except sqlite3.Error as e:
#         logger.error(f"Failed to initialize database: {e}")
#         raise


# def insert_feedback(
#     timestamp: str,
#     current_basis_weight: float,
#     predicted_basis_weight: float,
#     suggested_stock_flow: float,
#     suggested_steam_pressure: float,
#     operator_decision: str,
# ) -> int:
#     """
#     Inserts a single feedback record into the feedback_logs table.

#     Parameters
#     ----------
#     timestamp               : ISO-format string of when the event occurred.
#     current_basis_weight    : The actual basis weight reading at decision time.
#     predicted_basis_weight  : The XGBoost model's predicted basis weight.
#     suggested_stock_flow    : AI-recommended stock flow setpoint.
#     suggested_steam_pressure: AI-recommended steam pressure setpoint.
#     operator_decision       : 'Accepted' or 'Rejected'.

#     Returns
#     -------
#     int : The row ID of the newly inserted record.
#     """
#     if operator_decision not in ("Accepted", "Rejected"):
#         raise ValueError(f"operator_decision must be 'Accepted' or 'Rejected', got: {operator_decision!r}")

#     insert_sql = """
#     INSERT INTO feedback_logs (
#         timestamp,
#         current_basis_weight,
#         predicted_basis_weight,
#         suggested_stock_flow,
#         suggested_steam_pressure,
#         operator_decision
#     ) VALUES (?, ?, ?, ?, ?, ?);
#     """
#     try:
#         with get_connection() as conn:
#             cursor = conn.execute(
#                 insert_sql,
#                 (
#                     timestamp,
#                     current_basis_weight,
#                     predicted_basis_weight,
#                     suggested_stock_flow,
#                     suggested_steam_pressure,
#                     operator_decision,
#                 ),
#             )
#             conn.commit()
#             row_id = cursor.lastrowid
#         logger.info(f"Feedback logged successfully. Row ID={row_id}, Decision={operator_decision}")
#         return row_id
#     except sqlite3.Error as e:
#         logger.error(f"Failed to insert feedback: {e}")
#         raise


# def fetch_all_feedback() -> list[dict]:
#     """
#     Retrieves all feedback records from the database, ordered newest first.

#     Returns
#     -------
#     list[dict] : A list of dictionaries, one per feedback record.
#     """
#     select_sql = "SELECT * FROM feedback_logs ORDER BY id DESC;"
#     try:
#         with get_connection() as conn:
#             rows = conn.execute(select_sql).fetchall()
#         return [dict(row) for row in rows]
#     except sqlite3.Error as e:
#         logger.error(f"Failed to fetch feedback: {e}")
#         return []


# def get_feedback_summary() -> dict:
#     """
#     Returns a summary of accepted vs. rejected decisions.

#     Returns
#     -------
#     dict : {'total': int, 'accepted': int, 'rejected': int}
#     """
#     summary_sql = """
#     SELECT
#         COUNT(*) AS total,
#         SUM(CASE WHEN operator_decision = 'Accepted' THEN 1 ELSE 0 END) AS accepted,
#         SUM(CASE WHEN operator_decision = 'Rejected' THEN 1 ELSE 0 END) AS rejected
#     FROM feedback_logs;
#     """
#     try:
#         with get_connection() as conn:
#             row = conn.execute(summary_sql).fetchone()
#         return {
#             "total":    row["total"]    or 0,
#             "accepted": row["accepted"] or 0,
#             "rejected": row["rejected"] or 0,
#         }
#     except sqlite3.Error as e:
#         logger.error(f"Failed to fetch summary: {e}")
#         return {"total": 0, "accepted": 0, "rejected": 0}


# # --- Auto-initialize when module is imported ---
# initialize_database()



import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "operator_feedback.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            current_bw REAL,
            predicted_bw REAL,
            target_bw REAL,
            rec_stock_flow REAL,
            rec_steam_pressure REAL,
            operator_action TEXT,
            model_version TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_feedback(current_bw, predicted_bw, target_bw, rec_stock, rec_steam, action):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback_logs 
        (timestamp, current_bw, predicted_bw, target_bw, rec_stock_flow, rec_steam_pressure, operator_action, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        float(current_bw), float(predicted_bw), float(target_bw),
        float(rec_stock), float(rec_steam), action, "v2.1-XGB-Surrogate"
    ))
    conn.commit()
    conn.close()

def get_feedback_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM feedback_logs ORDER BY id DESC", conn)
    conn.close()
    return df

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")