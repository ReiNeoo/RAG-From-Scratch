import asyncio
# from flask import Flask, jsonify, Request
from fastapi import FastAPI, Depends, requests, File, UploadFile
from fastapi.responses import StreamingResponse

from src.rag_system import RAG
from src.chroma_database import VectorDB


app = FastAPI()


@app.get('query/get')
async def get_query(query):
    pass


@app.post('file')
async def post_file():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
