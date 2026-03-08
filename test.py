from utils.sqlite_db import SQLite_SqlDB

sql_db = SQLite_SqlDB()
print(sql_db.queryDataFromTable("Select * from persons"))
print(sql_db.queryDataFromTable("Select * from face_embeddings_id"))
