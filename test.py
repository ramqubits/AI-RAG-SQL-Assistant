import sqlite3

conn = sqlite3.connect("sql_assistant.db")
cursor = conn.cursor()
cursor.execute("SELECT id, question, final_answer, created_at FROM conversation_history ORDER BY id DESC LIMIT 5")
for row in cursor.fetchall():
    print(row)
conn.close()