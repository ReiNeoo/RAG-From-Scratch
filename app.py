import asyncio
import os

from typing import List
from fastapi import FastAPI, Depends, requests, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path

from src.rag_system import RAG


UPLOAD_DIRECTORY = Path("/home/proven/huggingface_model/data")

rag_system = RAG()
app = FastAPI()


@app.get('/query')
async def get_query(query):

    async def token_streamer():
        async for token in rag_system.get_response_tokens(query):
            await asyncio.sleep(0.1)
            yield token
    return StreamingResponse(token_streamer(), media_type="text/plain")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    rag_system.append_docs(file.filename)

    return JSONResponse(content={
        "filename": file.filename,
        "content_type": file.content_type,
        "file_path": file_location
    })


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        os.makedirs("data", exist_ok=True)
        file_location = f"data/{file.filename}"
        file.file.seek(0)

        with open(file_location, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        rag_system.append_docs(file.filename)
        results.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "file_path": file_location
        })

    return JSONResponse(content={"uploaded_files": results})


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
