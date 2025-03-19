from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq


class QAChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGroq(model="llama3-70b-8192", temperature=1)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever
        )

    def run(self, query):
        return self.qa_chain.run(query)
