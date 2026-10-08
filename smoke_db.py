#!/usr/bin/env python
"""Smoke test: connect to central SN68 MySQL from this box and return row counts.

Reads credentials from /root/.sn68_mysql_creds (never prints them).
Prints only aggregate counts.
"""
import os
import MySQLdb


def load_creds(path="/root/.sn68_mysql_creds"):
    for line in open(path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v.strip()


def main():
    load_creds()
    conn = MySQLdb.connect(
        host=os.environ["SN68_MYSQL_HOST"],
        port=int(os.environ["SN68_MYSQL_PORT"]),
        user=os.environ["SN68_MYSQL_USER"],
        password=os.environ["SN68_MYSQL_PASSWORD"],
        db=os.environ["SN68_MYSQL_DB"],
        charset="utf8mb4",
        compress=True,
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM observations")
    print("observations_rows=", cur.fetchone()[0])
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s",
        (os.environ["SN68_MYSQL_DB"],),
    )
    print("table_count=", cur.fetchone()[0])
    conn.close()
    print("DB_SMOKE_OK")


if __name__ == "__main__":
    main()
