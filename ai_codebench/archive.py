import sqlite3
import os
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta


class AnswerArchive:
    def __init__(self, retention_days=14, db_path="answers/answers.db"):
        self.retention_days = retention_days
        self.db_path = db_path
        self.lock = threading.Lock()  # For thread safety
        self.running = False
        self.thread = None
        self._init_db()

    def _init_db(self):
        # Create answers directory if it doesn't exist
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS archived_answers (
                    id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                )
            """
            )
            conn.commit()

    def archive_old_answers(self):
        with self.lock:
            answers_dir = Path("answers")
            if not answers_dir.exists():
                return

            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_date_str = cutoff_date.strftime("%Y%m%d")

            for date_dir in answers_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                # Check if directory name is a valid date
                if len(date_dir.name) != 8 or not date_dir.name.isdigit():
                    continue

                # Archive if directory date is older than cutoff
                if date_dir.name < cutoff_date_str:
                    for file_path in date_dir.glob("*.md"):
                        self._archive_file(date_dir.name, file_path)
                        # Remove file after archiving
                        os.remove(file_path)

                    # Remove empty directory
                    if not any(date_dir.iterdir()):
                        os.rmdir(date_dir)

    def _archive_file(self, date, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse question and answer from file content
            if content.startswith("Q:\n"):
                parts = content.split("\n\nA:\n", 1)
                question = parts[0][3:].strip() if len(parts) > 0 else ""
                answer = parts[1].strip() if len(parts) > 1 else ""
            else:
                question = ""
                answer = content.strip()

            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO archived_answers (date, filename, question, answer)
                    VALUES (?, ?, ?, ?)
                """,
                    (date, file_path.name, question, answer),
                )
                conn.commit()

        except Exception as e:
            print(f"Error archiving {file_path}: {str(e)}")

    def search_answers(self, query):
        results = []
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT id, date, filename, question, answer 
                        FROM archived_answers 
                        WHERE question LIKE ?
                    """,
                        (f"%{query}%",),
                    )
                    results = cursor.fetchall()
        except Exception as e:
            print(f"Search error: {str(e)}")
        return results

    def delete_answers(self, answer_ids):
        """Delete answers by their IDs atomically"""
        if not answer_ids:
            return

        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # Use parameterized query for safety
                    placeholders = ",".join("?" for _ in answer_ids)
                    cursor.execute(
                        f"DELETE FROM archived_answers WHERE id IN ({placeholders})",
                        answer_ids,
                    )
                    conn.commit()
        except Exception as e:
            print(f"Delete error: {str(e)}")

    def start_periodic_archiving(self, interval=86400):  # Default: 24 hours
        self.running = True

        def run():
            while self.running:
                self.archive_old_answers()
                time.sleep(interval)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
