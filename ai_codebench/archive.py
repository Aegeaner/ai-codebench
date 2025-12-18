import sqlite3
import os
import threading
from pathlib import Path
from datetime import datetime, timedelta


class AnswerArchive:
    def __init__(self, retention_days=14, db_path="answers/answers.db"):
        self.retention_days = retention_days
        self.db_path = db_path
        self.lock = threading.Lock()  # For thread safety
        self._stop_event = threading.Event()
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
            
            files_to_archive = []

            # First pass: Collect all files to archive
            for date_dir in answers_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                # Check if directory name is a valid date
                if len(date_dir.name) != 8 or not date_dir.name.isdigit():
                    continue

                # Archive if directory date is older than cutoff
                if date_dir.name < cutoff_date_str:
                    for file_path in date_dir.glob("*.md"):
                        files_to_archive.append((date_dir.name, file_path))

            if not files_to_archive:
                return

            # Second pass: Batch insert into database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for date, file_path in files_to_archive:
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

                            cursor.execute(
                                """
                                INSERT INTO archived_answers (date, filename, question, answer)
                                VALUES (?, ?, ?, ?)
                                """,
                                (date, file_path.name, question, answer),
                            )
                            
                            # Remove file after successful processing
                            # We'll remove files after commit to be safe, but here we are in a transaction.
                            # If we want to be strictly atomic, we should only delete if commit succeeds.
                            # For simplicity/performance in this CLI tool, we can delete after we confirm processing logic passed,
                            # but keeping track to delete later is safer.
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
                            # Skip this file but continue with others? 
                            # If transaction fails, we shouldn't delete any files.
                            pass

                    conn.commit()
                    
                    # Delete files only after successful commit
                    for _, file_path in files_to_archive:
                         try:
                             if file_path.exists():
                                 os.remove(file_path)
                         except OSError:
                             pass

                    # Clean up empty directories
                    for date_dir in answers_dir.iterdir():
                        if date_dir.is_dir() and date_dir.name < cutoff_date_str:
                             if not any(date_dir.iterdir()):
                                 try:
                                     os.rmdir(date_dir)
                                 except OSError:
                                     pass

            except Exception as e:
                print(f"Database error during archiving: {str(e)}")

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
        self._stop_event.clear()

        def run():
            while not self._stop_event.is_set():
                self.archive_old_answers()
                # Wait for interval or until stopped
                self._stop_event.wait(interval)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        if self.thread:
            self.thread.join()
