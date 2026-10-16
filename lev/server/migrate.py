"""Explicit offline v1 -> v2 migration; stop API and workers before invoking."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3

from lev.server.metrics import UPGRADE


def migrate(path, backup, *, _after_statement=None):
    path, backup = Path(path), Path(backup)
    if not path.is_file() or path.resolve() == backup.resolve():
        raise ValueError("an existing database and a separate new backup path are required")
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        if db.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("migration requires schema v1")
        # O_EXCL protects an earlier backup. SQLite backup includes committed WAL.
        fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        with closing(sqlite3.connect(backup)) as target:
            db.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("backup integrity check failed")
        with open(backup, "rb") as file:
            os.fsync(file.fileno())
        directory = os.open(backup.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN EXCLUSIVE")
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise RuntimeError("database changed during backup")
            for statement in UPGRADE.split(";"):
                if statement.strip():
                    db.execute(statement)
                    if _after_statement is not None:
                        _after_statement(statement)
            if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("foreign key check failed")
            db.commit()
        except BaseException:
            db.rollback()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--backup", required=True)
    args = parser.parse_args()
    migrate(args.db, args.backup)


if __name__ == "__main__":
    main()
