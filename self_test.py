from utils.faiss_db import FAISS_VectorDB
from utils.sqlite_db import SQLite_SqlDB

sql_db = SQLite_SqlDB()
vector_db = FAISS_VectorDB()

image_path = input("Image Path : ")

similarities, ids = vector_db.retrieveData(image_path)

results = []
print(f"Top 5 matches:")
print("-" * 60)

for i, (similarity, faiss_id) in enumerate(zip(similarities[0], ids[0]), 1):
    if faiss_id == -1:  # No match found
        continue
    
    data = sql_db.retrievePersonData(faiss_id)
    
    if data:
        name = data[0]
        description = data[1]
        results.append({
            "rank": i,
            "name": name,
            "description": description,
            "similarity": float(similarity)
        })
        print(f"{i}. {name}, Description : {description}, Similarity: {similarity:.2f}")

'''
similarities: numpy.ndarray of shape (1, k) containing float32 values
For k=5: shape is (1, 5)
Contains similarity scores (higher = better match for IndexFlatIP/cosine similarity)

ids: numpy.ndarray of shape (1, k) containing int64 values
For k=5: shape is (1, 5)
Contains the FAISS IDs of the matched faces
Value is -1 if no match found

Eg.
similarities = [[0.95, 0.87, 0.82, 0.75, 0.68]]  # numpy array, dtype=float32
ids = [[6, 2, 5, 1, 9]]  # numpy array, dtype=int64

'''

'''
for i, (similarity, faiss_id) in enumerate(zip(similarities[0], ids[0]), 1):

This line iterates through the matched results and unpacks them. Here's what each part does:

Breaking it down:

similarities[0] - Extracts the first row from the 2D array
[[0.95, 0.87, 0.82, 0.75, 0.68]] → [0.95, 0.87, 0.82, 0.75, 0.68]

ids[0] - Extracts the first row from the 2D array
[[6, 2, 5, 1, 9]] → [6, 2, 5, 1, 9]

zip(similarities[0], ids[0]) - Pairs corresponding elements together
Creates: (0.95, 6), (0.87, 2), (0.82, 5), (0.75, 1), (0.68, 9)

enumerate(..., 1) - Adds a counter starting from 1 (instead of default 0)
Creates: (1, (0.95, 6)), (2, (0.87, 2)), (3, (0.82, 5)), ...

i, (similarity, faiss_id) - Unpacks each iteration into:
i = rank number (1, 2, 3, 4, 5)
similarity = similarity score (0.95, 0.87, etc.)
faiss_id = FAISS ID (6, 2, 5, etc.)

Example iteration:
1st loop: i=1, similarity=0.95, faiss_id=6
2nd loop: i=2, similarity=0.87, faiss_id=2
3rd loop: i=3, similarity=0.82, faiss_id=5


'''