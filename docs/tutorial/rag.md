# Build Your First RAG System with RAGEngine

In EvoAgentX, the `RAGEngine` is a powerful tool for building Retrieval-Augmented Generation (RAG) systems. It allows you to load documents, create searchable indices, and retrieve relevant information to answer questions. This tutorial is designed for beginners and will guide you through the essential steps to create and use a RAG system with `RAGEngine`:

1. **Setting Up RAGEngine**: Learn how to configure and initialize the engine.
2. **Indexing and Querying Documents**: Load a simple dataset and query it to find answers.
3. **Saving and Loading Indices**: Save your indexed data and reuse it later.

By the end of this tutorial, you’ll be able to set up a basic RAG system, index documents, query them, and persist your work for future use. We’ll use a sample from the HotPotQA dataset to make it practical and fun!

## 1. Setting Up RAGEngine

The first step is to set up your environment and initialize `RAGEngine`. This involves installing dependencies, configuring the storage backend, and setting up the embedding model.

### Install Dependencies
Ensure you have EvoAgentX installed. You’ll also need an OpenAI API key for embeddings. Run the following in your terminal:

```bash
pip install evoagentx llama_index pydantic python-dotenv
```

Create a `.env` file in your project directory and add your OpenAI API key:

```plaintext
OPENAI_API_KEY=your-openai-api-key
```

### Configure the Environment
Let’s write a Python script to set up `RAGEngine`. We’ll use SQLite for metadata storage and FAISS for vector embeddings, which are beginner-friendly options.

```python
import os
from dotenv import load_dotenv
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure storage (SQLite for metadata, FAISS for vectors)
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Configure RAGEngine
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.5)
)

# Initialize RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("RAGEngine is ready to go!")
```

### What’s Happening Here?
- **Environment Setup**: We load the OpenAI API key from the `.env` file.
- **Storage Configuration**: We set up SQLite to store metadata and FAISS to store vector embeddings. The `dimensions=1536` matches the OpenAI embedding model.
- **RAG Configuration**: We configure the pipeline:
  - `ReaderConfig`: Reads files (we’ll use a JSON file later).
  - `ChunkerConfig`: Splits documents into 512-character chunks with 50-character overlap.
  - `EmbeddingConfig`: Uses OpenAI’s `text-embedding-ada-002` for embeddings.
  - `IndexConfig`: Creates a vector index.
  - `RetrievalConfig`: Retrieves the top 3 most similar chunks with a similarity score above 0.5.
- **Initialization**: We create the `RAGEngine` instance, ready to process documents.

Save this code as `rag_setup.py` and run it:

```bash
python rag_setup.py
```

If you see “RAGEngine is ready to go!”, you’re set! If you encounter errors, check your API key or ensure dependencies are installed.

For more details on configuration, see the [RAGEngine documentation](../modules/rag.md).

## 2. Indexing and Querying Documents

Now, let’s index a document from the HotPotQA dataset and query it. We’ll use the first HotPotQA example you provided, which contains information about Scott Derrickson and Ed Wood, and answer the question: “Were Scott Derrickson and Ed Wood of the same nationality?”

### Prepare the Dataset
Create a directory called `data` and save the HotPotQA example as `hotpotqa_sample.json`:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "answer": "yes",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ],
  "context": [
    ["Ed Wood (film)", [
      "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.",
      "The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau.",
      "Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast."
    ]],
    ["Scott Derrickson", [
      "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
      "He lives in Los Angeles, California.",
      "He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""
    ]],
    ["Ed Wood", [
      "Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
    ]]
  ]
}
```

For simplicity, we’ve trimmed the context to include only the relevant entries about Scott Derrickson and Ed Wood.

### Index the Document
Let’s write a script to load the JSON file, index its content, and query it. Create a file called `rag_tutorial.py`:

```python
import json
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.schema import Query
# Import setup code from earlier
from rag_setup import rag_engine

# Step 1: Load and index the HotPotQA sample
with open("./data/hotpotqa_sample.json", "r", encoding="utf-8") as f:
    hotpotqa_data = json.load(f)

# Create a temporary text file with the context for indexing
context = hotpotqa_data["context"]
with open("./data/hotpotqa_context.txt", "w", encoding="utf-8") as f:
    for title, sentences in context:
        f.write(f"# {title}\n")
        for sentence in sentences:
            f.write(f"{sentence}\n\n")

# Index the text file
corpus = rag_engine.read(
    file_paths="./data/hotpotqa_context.txt",
    filter_file_by_suffix=[".txt"],
    merge_by_file=True,
    show_progress=True,
    corpus_id="hotpotqa_corpus"
)
rag_engine.add(index_type="vector", nodes=corpus, corpus_id="hotpotqa_corpus")

print("Documents indexed successfully!")

# Step 2: Query the index
query = Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?", top_k=3)
result = rag_engine.query(query, corpus_id="hotpotqa_corpus")

# Print the retrieved chunks
print("\nRetrieved answers:")
for i, chunk in enumerate(result.corpus.chunks, 1):
    print(f"{i}. {chunk.text}")

# Clean up
rag_engine.clear(corpus_id="hotpotqa_corpus")
```

### Run the Script
Run the script:

```bash
python rag_tutorial.py
```

### What’s Happening Here?
- **Loading the Data**: We read the HotPotQA JSON file and extract its `context` field, which contains text about Scott Derrickson and Ed Wood.
- **Indexing**:
  - We create a temporary text file (`hotpotqa_context.txt`) with the context, formatted for readability.
  - The `read` method loads the text file into a `Corpus`, splitting it into chunks (based on the 512-character limit set earlier).
  - The `add` method indexes the chunks into a vector index using OpenAI embeddings.
- **Querying**:
  - We create a `Query` object with the question from the dataset.
  - The `query` method retrieves the top 3 chunks most similar to the question.
  - The results include sentences indicating both individuals are American (e.g., “Scott Derrickson… is an American director” and “Edward Davis Wood Jr… was an American filmmaker”).
- **Cleanup**: We clear the index to free up memory.

### Expected Output
You should see something like:

```
Documents indexed successfully!

Retrieved answers:
1. Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
2. Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.
3. He lives in Los Angeles, California.
```

These chunks confirm that both Scott Derrickson and Ed Wood are American, answering the question “yes.”

For more details on indexing and querying, see the [RAGEngine documentation](../modules/rag.md).

## 3. Saving and Loading Indices

Once you’ve indexed your documents, you can save the index to reuse it later without reprocessing the data. This is useful for large datasets or production environments.

### Save the Index
Modify the `rag_tutorial.py` script to save the index after indexing. Add the following line after `rag_engine.add`:

```python
# Save the index to disk
rag_engine.save(output_path="./data/indexing", corpus_id="hotpotqa_corpus", index_type="vector")
print("Index saved to ./data/indexing")
```

### Load the Index
To load the saved index and query it, create a new script called `rag_load.py`:

```python
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.schema import Query
from rag_setup import rag_engine

# Load the saved index
rag_engine.load(source="./data/indexing", corpus_id="hotpotqa_corpus", index_type="vector")
print("Index loaded successfully!")

# Query the loaded index
query = Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?", top_k=3)
result = rag_engine.query(query, corpus_id="hotpotqa_corpus")

# Print the retrieved chunks
print("\nRetrieved answers:")
for i, chunk in enumerate(result.corpus.chunks, 1):
    print(f"{i}. {chunk.text}")

# Clean up
rag_engine.clear(corpus_id="hotpotqa_corpus")
```

### Run the Script
Run the script:

```bash
python rag_load.py
```

### What’s Happening Here?
- **Saving**: The `save` method stores the vector index and metadata to the `./data/indexing` directory.
- **Loading**: The `load` method reconstructs the index from the saved files, making it ready for querying without re-indexing.
- **Querying**: We run the same query as before, and the results should be identical.
- **Cleanup**: We clear the index to keep things tidy.

### Expected Output
The output should be the same as in Section 2, confirming that the loaded index works correctly.

For more details on saving and loading, see the [RAGEngine documentation](../modules/rag.md).

## Next Steps
Congratulations! You’ve built your first RAG system with `RAGEngine`. Here are some ideas to explore next:
- Try indexing a larger dataset or different file types (e.g., PDFs).
- Experiment with different chunk sizes or embedding models (e.g., Hugging Face).
- Integrate `RAGEngine` with an EvoAgentX agent to answer questions automatically.

For a complete example, refer to the [RAGEngine example](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/rag_engine.py).

Happy building with EvoAgentX!