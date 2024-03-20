import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load the CSV file into a DataFrame
df = pd.read_csv('books.csv')

# Extract book descriptions
book_descriptions = df['Description']

# Encode book descriptions using SentenceTransformer
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(book_descriptions)

# Initialize Faiss index
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

query_description = input("Enter Search Query: ")
# Perform a search in Faiss index
query_vector = encoder.encode([query_description])  # Encode the query description
faiss.normalize_L2(query_vector)

k = 3  # Number of nearest neighbors to retrieve
D, I = index.search(query_vector, k)

# Retrieve titles and similarity scores corresponding to the search results
similar_books = []
for d, idx in zip(D[0], I[0]):
    title = df.iloc[idx]['Title']  # Get the title using the index
    similar_books.append((title, d))  # Append title and similarity score

# Print the titles and similarity scores of similar books
print("Similar books:")
for title, score in similar_books:
    print(f"Title: {title}, Similarity Score: {score}")

