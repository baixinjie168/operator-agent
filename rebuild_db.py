import sqlite3

conn = sqlite3.connect("data/operator_agent.db")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Existing tables:", tables)
for t in tables:
    if t.startswith("sqlite_"):
        continue
    conn.execute(f"DROP TABLE IF EXISTS [{t}]")
conn.commit()

indexes = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'").fetchall()]
for i in indexes:
    conn.execute(f"DROP INDEX IF EXISTS [{i}]")
conn.commit()

schema = open("packages/mcp-server/src/mcp_server/schema.sql").read()
conn.executescript(schema)
conn.commit()

new_tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
conn.close()
print("Rebuilt tables:", new_tables)