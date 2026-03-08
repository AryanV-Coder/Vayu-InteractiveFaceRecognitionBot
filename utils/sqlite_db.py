import sqlite3

class SQLite_SqlDB():
    def __init__(self, db_path="face_database.db"):

        self.db_path = db_path
        
        # Initialize SQLite database
        self.init_sqlite_db()
    
    def init_sqlite_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # CRITICAL: Enable foreign key constraints for this connection
        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings_id (
                faiss_id INTEGER PRIMARY KEY,
                person_id INTEGER NOT NULL,
                FOREIGN KEY (person_id) 
                    REFERENCES persons (person_id) 
                    ON DELETE CASCADE
                    ON UPDATE CASCADE
            );
        ''')

        conn.commit()
        cursor.close()
        conn.close()
    
        print(f" ✓ Database '{self.db_path}' initialized.")
    
    def retrievePersonData(self, faiss_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Best practice: always enable PRAGMA before querying if you rely on relationships
        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute('''
            SELECT persons.name, persons.description 
            FROM persons 
            JOIN face_embeddings_id ON persons.person_id = face_embeddings_id.person_id 
            WHERE face_embeddings_id.faiss_id = ?;
        ''',(faiss_id,))

        result = cursor.fetchone()
        
        cursor.close()
        conn.close()

        # result will be a tuple like ("Rahul", "Loves guitar") or None
        if result:
            print(f" ✓ Match Found: {result[0]}")
            return result
        else:
            print(f" ✗ No match found for FAISS ID: {faiss_id}")
            return None

    def queryDataFromTable(self, query):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Best practice: always enable PRAGMA before querying if you rely on relationships
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        cursor.execute(query)

        result = cursor.fetchall()
        
        cursor.close()
        conn.close()

        if result:
            print(" ✓ Data Found")
            return result
        else:
            print(" ✗ No Data Found")
            return None
    
    def insertDataIntoTable(self, faiss_ids : list, name : str, description : str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_keys = ON;")

        # Insert person and get their ID
        cursor.execute("INSERT INTO persons(name, description) VALUES (?, ?)", 
                    (name, description))
        person_id = cursor.lastrowid  # Get the auto-generated person_id
        
        # Pair each faiss_id with the same person_id
        data = [(faiss_id, person_id) for faiss_id in faiss_ids]
        
        # Insert all face_embeddings_id entries
        cursor.executemany(
            "INSERT INTO face_embeddings_id (faiss_id, person_id) VALUES (?, ?)",
            data
        )
        
        conn.commit()
        cursor.close()
        conn.close()

        print(f"✓ Added {name} to SQLite database")

    def maxFaissId(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute("SELECT MAX(faiss_id) FROM face_embeddings_id")

        result = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        if result:
            return result
        else :
            return 0
