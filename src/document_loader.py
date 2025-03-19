import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, data_path, chunk_size=50, chunk_overlap=10):
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
