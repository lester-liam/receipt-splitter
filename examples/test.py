import httpx
import time
import pprint

def extract_img(url, filename: str) -> None:
    """
    Test the /extract endpoint with an image file.
    Sends a multipart file upload to the FastAPI endpoint.
    """
    file_path = f"./img/{filename}"
    
    with open(file_path, "rb") as f:
        files = {"image": (filename, f, "image/jpeg")}
        
        start_time = time.time()
        response = httpx.post(url, files=files, timeout=180)
        end_time = time.time()
        
        print(f"Response Code: [{response.status_code}]")
        print(f"Processing {filename} took {end_time - start_time:.2f} seconds.")
        
        response.raise_for_status()
        result = response.json()
        
        print(f"Items extracted: {len(result.get('items', []))}")
        print(f"SST: {result.get('sst')}")
        print(f"Service Charge: {result.get('serviceCharge')}")
        pprint.pp(f"Response: {result}")


if __name__ == "__main__":
    url = "http://localhost:8000/extract"
    filename = "4.jpeg"
    extract_img(url, filename)