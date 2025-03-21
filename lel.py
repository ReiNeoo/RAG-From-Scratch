from src.rag_system import RAG
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.qa_chain import QAChain
from src.document_loader import DocumentLoader
from src.chroma_database import VectorDB
import asyncio
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_community.vectorstores import Chroma
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, data_path, chunk_size=500, chunk_overlap=100):
        self.path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def load_documents(self):
        for file in os.listdir(path=self.path):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.path, file)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                texts = self.splitter.split_documents(docs)
                yield texts
            elif file.endswith(".txt"):
                file_path = os.path.join(self.path, file)
                loader = TextLoader(file_path)
                docs = loader.load()
                texts = self.splitter.split_documents(docs)
                yield texts
            elif file.endswith(".docx"):
                file_path = os.path.join(self.path, file)
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                texts = self.splitter.split_documents(docs)
                yield texts


class VectorDB:
    def __init__(self, embedding, persist_directory):
        self.persist_directory = persist_directory
        self.embedding = embedding
        self.database = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding)

    def create_vector_store(self, documents):
        if self.database._collection.count() == 0:
            self.database = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_directory)
        else:
            self.append_vectors(documents)

        self.database.persist()

    def retriever(self):
        return self.database.as_retriever()

    def append_vectors(self, documents):
        try:
            self.database.add_documents(documents)
            self.database.persist()
        except Exception as e:
            print(f'ERROR: {e}')

    def is_empty(self):
        return self.database._collection.count() == 0


class StremingCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.tokens = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.tokens.put(token)

    async def on_llm_end(self, **kwargs):
        await self.tokens.put("[DONE]")


class QAChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.callback_handler = StremingCallbackHandler()
        self.llm = ChatGroq(model="llama3-70b-8192",
                            temperature=1, streaming=True)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            callbacks=[self.callback_handler]
        )

    async def invoke(self, question):
        task = asyncio.create_task(self.qa_chain.ainvoke({"query": question}))
        while True:
            try:
                token = await asyncio.wait_for(
                    self.callback_handler.tokens.get(), timeout=10.0)
                print(token)
                if token == "[DONE]":
                    break

                if task.done():
                    print("task done")
                yield token

            except asyncio.TimeoutError:
                if task.done():
                    print("Timeout")
                    break


PATH = "/home/proven/huggingface_model/data"
DB_PATH = "/home/proven/huggingface_model/chroma_db"


class RAG:
    def __init__(self):
        self.doc_loader = DocumentLoader(PATH)
        self.embedgings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.documents = self._load_documents()
        self.database = VectorDB(self.embedgings, DB_PATH)
        self.database.create_vector_store(self.documents)
        self.qa_chain = QAChain(self.database.retriever())

    def _load_documents(self):
        documents = []
        for chunked_doc in self.doc_loader.load_documents():
            documents.extend(chunked_doc)
        return documents

    def _vectorize_documents(self, doc):
        self.database.create_vector_store(doc)

    async def get_response_tokens(self, query):
        async for token in self.qa_chain.invoke(query):
            yield token

    def get_query_generator(self, query):
        pass


async def main():
    rag_system = RAG()
    query = "Gas fee olacak mı? Olmayacaksa işlemler nasıl gerçekleşmektedir?"

    async for token in rag_system.get_response_tokens(query):
        print(token)


if __name__ == "__main__":
    asyncio.run(main())
