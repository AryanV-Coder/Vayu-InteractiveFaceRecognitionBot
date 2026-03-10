from utils.faiss_db import FAISS_VectorDB
from utils.sqlite_db import SQLite_SqlDB

class FaceRecognitionLogic():
    def __init__(self):
        self.sql_db = SQLite_SqlDB()
        self.vector_db = FAISS_VectorDB()

    def recognise_face(self,image_path):
        similarities, ids = self.vector_db.retrieveData(image_path,k=1)
        faiss_id = ids[0][0]
        similarity_score = similarities[0][0]

        if faiss_id == -1:  # No match found
            print(" ❌ No Match Found in Vector DB")
            return None

        # Apply threshold check
        if similarity_score < 0.65:
            print(f" ⚠️ Low Similarity ({similarity_score:.2f} < 0.65) - Defaulting to Unknown")
            return {
                "name" : "Unknown",
                "description" : ""
            }

        data = self.sql_db.retrievePersonData(faiss_id)

        if not data:
            print(f" ❌ No Match Found in Sql DB for FAISS ID : {faiss_id}")
            return None

        name = data[0]
        description = data[1]
        
        print(f" ✅ Name : {name}, Description : {description}, Similarity: {similarity_score:.2f}")

        return {
            "name" : name,
            "description" : description
        }