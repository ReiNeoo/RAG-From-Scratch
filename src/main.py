from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PATH = "/home/proven/huggingface_model/data/example.data"

# loader = TextLoader(PATH)
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100,
# )

# texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# database = Chroma.from_documents(
#     texts, embeddings, persist_directory="./chroma_db")

print("Documents vectorized and saved successfully!")

# query = "What is AI?"
# results = database.similarity_search(query, k=3)  # Return top 3 results

# for i, result in enumerate(results):
#     print(f"Result {i+1}:\n{result.page_content}\n")

# database.persist()
