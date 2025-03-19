from langchain_community.vectorstores import Chroma


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
