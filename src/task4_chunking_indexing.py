from pathlib import Path



STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"





# =============================================================================

# CONFIGURATION

# =============================================================================



# RecursiveCharacterTextSplitter phù hợp với corpus IELTS vì có thể chia văn bản

# theo nhiều cấp độ: đoạn -> dòng -> câu -> từ, giúp hạn chế việc cắt giữa

# các câu hoặc đoạn có ý nghĩa.

#

# 500 characters giúp chunk đủ nhỏ để retrieval tìm được đoạn liên quan,

# nhưng vẫn giữ đủ ngữ cảnh của tiêu chí chấm điểm và essay mẫu.

#

# Overlap 50 characters giúp giữ lại một phần ngữ cảnh giữa hai chunk liên tiếp,

# hạn chế mất thông tin khi nội dung nằm ở ranh giới giữa các chunk.



CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

CHUNKING_METHOD = "recursive"





# BGE-M3 hỗ trợ multilingual, phù hợp với corpus có tiếng Anh và tiếng Việt.

# Embedding dimension của BGE-M3 là 1024.



EMBEDDING_MODEL = "BAAI/bge-m3"

EMBEDDING_DIM = 1024





# ChromaDB đơn giản, chạy local và hỗ trợ persistent storage,

# phù hợp với project RAG local.



VECTOR_STORE = "chromadb"

COLLECTION_NAME = "ielts_writing_docs"





# =============================================================================

# IMPLEMENTATION

# =============================================================================



def load_documents() -> list[dict]:

    """

    Đọc toàn bộ markdown files từ data/standardized/.



    Returns:

        List of {

            'content': str,

            'metadata': {

                'source': str,

                'type': str

            }

        }

    """

    documents = []



    for md_file in STANDARDIZED_DIR.rglob("*.md"):

        content = md_file.read_text(encoding="utf-8").strip()



        # Bỏ qua file markdown rỗng

        if not content:

            print(f"⚠ Skipping empty file: {md_file.name}")

            continue



        documents.append({

            "content": content,

            "metadata": {

                "source": md_file.name,

                "type": "ielts_writing"

            }

        })



    return documents





def chunk_documents(documents: list[dict]) -> list[dict]:

    """

    Chunk documents theo RecursiveCharacterTextSplitter.



    Returns:

        List of {

            'content': str,

            'metadata': dict

        }

    """

    from langchain_text_splitters import RecursiveCharacterTextSplitter



    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[

            "\n\n",

            "\n",

            ". ",

            " ",

            ""

        ]

    )



    chunks = []



    for doc in documents:

        splits = splitter.split_text(doc["content"])



        for i, chunk_text in enumerate(splits):

            chunks.append({

                "content": chunk_text,

                "metadata": {

                    **doc["metadata"],

                    "chunk_index": i

                }

            })



    return chunks





def embed_chunks(chunks: list[dict]) -> list[dict]:

    """

    Embed toàn bộ chunks bằng BAAI/bge-m3.

    """

    from sentence_transformers import SentenceTransformer



    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")



    model = SentenceTransformer(EMBEDDING_MODEL)



    texts = [chunk["content"] for chunk in chunks]



    embeddings = model.encode(

        texts,

        show_progress_bar=True,

        normalize_embeddings=True

    )



    for chunk, embedding in zip(chunks, embeddings):

        chunk["embedding"] = embedding.tolist()



    return chunks





def index_to_vectorstore(chunks: list[dict]):

    """

    Lưu chunks vào ChromaDB.

    """

    import chromadb



    CHROMA_DIR.mkdir(parents=True, exist_ok=True)



    client = chromadb.PersistentClient(

        path=str(CHROMA_DIR)

    )



    # Xóa collection cũ để tránh dữ liệu cũ lẫn với corpus mới.

    try:

        client.delete_collection(name=COLLECTION_NAME)

        print(f"✓ Deleted old collection: {COLLECTION_NAME}")

    except Exception:

        pass



    collection = client.get_or_create_collection(

        name=COLLECTION_NAME,

        metadata={

            "hnsw:space": "cosine"

        }

    )



    ids = [

        f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"

        for c in chunks

    ]



    collection.upsert(

        ids=ids,

        documents=[c["content"] for c in chunks],

        embeddings=[c["embedding"] for c in chunks],

        metadatas=[c["metadata"] for c in chunks]

    )



    print(f"✓ Collection: {COLLECTION_NAME}")

    print(f"✓ Total indexed chunks: {collection.count()}")

def get_embedding_model():

    """

    Load và trả về cùng embedding model được sử dụng ở Task 4.

    """

    from sentence_transformers import SentenceTransformer



    return SentenceTransformer(EMBEDDING_MODEL)





def get_collection():

    """

    Kết nối tới ChromaDB collection đã được tạo ở Task 4.

    """

    import chromadb



    client = chromadb.PersistentClient(

        path=str(CHROMA_DIR)

    )



    return client.get_collection(

        name=COLLECTION_NAME

    )



def run_pipeline():

    """Chạy toàn bộ pipeline: load → chunk → embed → index."""



    print("=" * 50)

    print("Task 4: Chunking & Indexing")

    print(f"  Chunking: {CHUNKING_METHOD}")

    print(f"  Chunk size: {CHUNK_SIZE}")

    print(f"  Chunk overlap: {CHUNK_OVERLAP}")

    print(f"  Embedding: {EMBEDDING_MODEL}")

    print(f"  Embedding dim: {EMBEDDING_DIM}")

    print(f"  Vector Store: {VECTOR_STORE}")

    print("=" * 50)



    docs = load_documents()

    print(f"\n✓ Loaded {len(docs)} documents")



    chunks = chunk_documents(docs)

    print(f"✓ Created {len(chunks)} chunks")



    chunks = embed_chunks(chunks)

    print(f"✓ Embedded {len(chunks)} chunks")



    index_to_vectorstore(chunks)



    print("\n✓ Task 4 completed successfully!")





if __name__ == "__main__":

    run_pipeline()