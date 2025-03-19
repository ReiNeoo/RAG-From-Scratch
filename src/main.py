from chroma_database import VectorDB
from document_loader import DocumentLoader
from qa_chain import QAChain

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

PATH = "/home/proven/huggingface_model/data"
DB_PATH = "/home/proven/huggingface_model/chroma_db"


def main():
    documents = []
    document_loader = DocumentLoader(PATH)
    for chunked_doc in document_loader.load_documents():
        documents.extend(chunked_doc)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    database = VectorDB(embeddings, DB_PATH)
    database.create_vector_store(documents, overwrite=True)
    qa_chain = QAChain(database.retriever())

    # query = "Proven Exchange üzerinden verilen tüm emirler veya yapılan swap işlemleri nerede görüntülenir?"
    query = "what is a blashumbabum"
    response = qa_chain.run(query)

    print(f"Response: {response}")


if __name__ == "__main__":
    main()
