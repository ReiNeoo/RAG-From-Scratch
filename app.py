import asyncio
import shutil
from fastapi import FastAPI, Depends, requests, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path

from src.rag_system import RAG
from src.chroma_database import VectorDB

from src.rag_system import RAG

UPLOAD_DIRECTORY = Path("/home/proven/huggingface_model/data")

rag_system = RAG()
app = FastAPI()


@app.get('/query')
async def get_query(query):

    print("geldik yoktunuz")

    async def token_streamer():
        async for token in StreamingResponse(query):
            yield token
    return StreamingResponse(token_streamer(), media_type="text/plain"), 200


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Optional: Save the file
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    rag_system.database.append_vectors(file_location)

    return JSONResponse(content={
        "filename": file.filename,
        "content_type": file.content_type,
        "file_path": file_location
    })


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
