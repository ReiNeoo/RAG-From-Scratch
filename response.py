import httpx
import asyncio


async def get_token_stream(url: str, params: dict):
    async with httpx.AsyncClient() as client:
        async with client.stream('GET', url) as response:
            # Check if request was successful
            if response.status_code == 200:
                async for chunk in response.aiter_text():
                    # Process each chunk (token)
                    print(chunk, end='', flush=True)
            else:
                print(f"Error: {response.status_code}")

# Example URL and parameters for an LLM API
query = "Gas fee olacak mı? Olmayacaksa işlemler nasıl gerçekleşmektedir?"
url = f"http://0.0.0.0:8000/query?query={query}"
params = {'prompt': 'Hello world', 'max_tokens': 100}

# Running the async function
asyncio.run(get_token_stream(url, params))
