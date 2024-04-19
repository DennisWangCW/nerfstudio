import sqlite3

# 连接到数据库
conn = sqlite3.connect('/workspace/trial_new/samples/chunk_0/colmap/database.db')
cursor = conn.cursor()

# 获取数据库中所有表格的名称
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# 遍历每个表格，并打印其字段名（即键）
for table in tables:
    table_name = table[0]
    print(f"Table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    print("Keys:", column_names)
    print()

    cursor.execute("SELECT prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images")
    rows = cursor.fetchall()

    # 打印结果
    for row in rows:
        print("prior_qw:", row[0])
        print("prior_qx:", row[1])
        print("prior_qy:", row[2])
        print("prior_qz:", row[3])
        print("prior_tx:", row[4])
        print("prior_ty:", row[5])
        print("prior_tz:", row[6])
        print("\n")

# 关闭连接
conn.close()

# 关闭数据库连接
conn.close()
