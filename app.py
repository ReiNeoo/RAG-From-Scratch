import asyncio
# from flask import Flask, jsonify, Request
from fastapi import FastAPI, Depends, requests, File, UploadFile
from fastapi.responses import StreamingResponse

from src.rag_system import RAG
from src.chroma_database import VectorDB

from src.rag_system import RAG

rag_system = RAG()
# query = "Gas fee olacak mı? Olmayacaksa işlemler nasıl gerçekleşmektedir?"

app = FastAPI()

# app.mount("/static")


@app.get('query/get')
async def get_query(query):
    return StreamingResponse(
        rag_system.get_response_tokens(query)
    )


@app.post('file')
async def post_file():
    pass


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
