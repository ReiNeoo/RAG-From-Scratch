from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
import asyncio


class StremingCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.tokens = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.tokens.put(token)

    async def on_llm_end(self, *args, **kwargs) -> None:
        await self.tokens.put("[DONE]")


class QAChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.callback_handler = StremingCallbackHandler()
        # self.llm = ChatGroq(model="llama3-70b-8192",
        #                     temperature=1, streaming=True, callbacks=[self.callback_handler])
        self.llm = OllamaLLM(
            model="llama3.1",
            streaming=True,
            temperature=1,
            callbacks=[self.callback_handler]
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            callbacks=[self.callback_handler]
        )

    async def invoke(self, question):
        task = asyncio.create_task(self.qa_chain.ainvoke({"query": question}))

        while True:
            try:
                token = await asyncio.wait_for(
                    self.callback_handler.tokens.get(), timeout=2.0)

                if token == "[DONE]":
                    break

                yield token

            except asyncio.TimeoutError:
                if task.done():
                    print("Timeout occurred while waiting for tokens")
                    break
