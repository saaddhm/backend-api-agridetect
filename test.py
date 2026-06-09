import sqlite3

conn = sqlite3.connect("agridetect_test.db")
cursor = conn.cursor()

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
""")

tables = cursor.fetchall()

for table in tables:
    table_name = table[0]

    print(f"\n=== {table_name} ===")

    cursor.execute(f"PRAGMA table_info({table_name})")

    for column in cursor.fetchall():
        print(column)

conn.close()