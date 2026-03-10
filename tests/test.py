import os
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sqlite_db import SQLite_SqlDB

sql_db = SQLite_SqlDB()
print(sql_db.queryDataFromTable("Select * from persons"))
print(sql_db.queryDataFromTable("Select * from face_embeddings_id"))
