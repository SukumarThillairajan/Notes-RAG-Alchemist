# test_embeddings.py

import os
from dotenv import load_dotenv

# --- CRITICAL ---
# Load environment variables from .env BEFORE importing your app code
# This makes sure the OPENAI_API_KEY is set.
load_dotenv()
# ----------------

# Now, import your functions
from utils.embeddings import get_embedding, batch_get_embeddings, EMBED_DIM

print("Starting embedding utility tests...\n")

# --- Test 1: Single Embedding ---
print("--- TEST 1: Single Embedding (get_embedding) ---")
try:
    text = "This is a test string."
    vector = get_embedding(text)
    
    print(f"Text: '{text}'")
    print(f"Vector Length: {len(vector)}")
    print(f"Vector preview: {vector[:5]}...") # Print first 5 dimensions
    
    # Check the dimension
    assert len(vector) == EMBED_DIM, f"Vector length is not {EMBED_DIM}!"
    print("STATUS: ✅ PASSED\n")

except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")


# --- Test 2: Batch Embeddings ---
print("--- TEST 2: Batch Embeddings (batch_get_embeddings) ---")
try:
    texts = [
        "First test string.",
        "Second test string."
    ]
    vectors = batch_get_embeddings(texts)
    
    print(f"Number of texts: {len(texts)}")
    print(f"Number of vectors returned: {len(vectors)}")
    
    # Check counts
    assert len(texts) == len(vectors), "Mismatch between input and output count!"
    
    # Check dimensions of each vector in the batch
    for i, vector in enumerate(vectors):
        print(f"  Vector {i} length: {len(vector)}")
        assert len(vector) == EMBED_DIM, f"Vector {i} length is not {EMBED_DIM}!"
    
    print("STATUS: ✅ PASSED\n")

except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")

print("...Embedding tests finished.")
