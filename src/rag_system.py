from src.chroma_database import VectorDB
from src.document_loader import DocumentLoader
from src.qa_chain import QAChain

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

PATH = "/home/proven/huggingface_model/data"
DB_PATH = "/home/proven/huggingface_model/chroma_db"


class RAG:
    def __init__(self):
        self.doc_loader = DocumentLoader(PATH)
        self.embedgings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.documents = self.load_plenty_of_documents()
        self.database = VectorDB(self.embedgings, DB_PATH)
        self.database.create_vector_store(self.documents)
        self.qa_chain = QAChain(self.database.retriever())

    def load_plenty_of_documents(self):
        documents = []
        for chunked_doc in self.doc_loader.load_documents():
            documents.extend(chunked_doc)
        return documents

    def append_docs(self, file):
        document = self.doc_loader.prepare_doc(file)
        self.database.append_vectors(document)

    def _vectorize_documents(self, doc):
        self.database.create_vector_store(doc)

    async def get_response_tokens(self, query):
        async for token in self.qa_chain.invoke(query):
            yield token

    def get_query_generator(self, query):
        pass
