from src.rag_system import RAG

import asyncio


async def main():
    rag_system = RAG()
    query = "Gas fee olacak mı? Olmayacaksa işlemler nasıl gerçekleşmektedir?"

    async for token in rag_system.get_response_tokens(query):
        print(token)


if __name__ == "__main__":
    asyncio.run(main())
