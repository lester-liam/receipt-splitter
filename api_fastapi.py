"""
FastAPI Endpoint for Receipt Extraction using Ollama/Gemini API

Pipeline/Process:
- Receive an image as bytes
- Run OCR (PaddleOCR) to detect/recognize text
- Parse Recognized Text into (Ollama/Gemini) Model for Field Extraction
- Extracts:
  - Mandatory: item name, price, quantity
  - Optional: SST %, service charge % (default 0)
- Coerce Response & Returns JSON matching ExtractReceiptResponse model:

# Sample JSON Response
{ 
  "items": [{"id": "...", "name": "...", "price": 0.0, "quantity": 1}, ...],
  "sst": 0,
  "serviceCharge": 0,
  "notes": "..."
}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4

import cv2
import httpx
import requests
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi_throttle import RateLimiter
from google import genai
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field

# Load Environment
load_dotenv()

# Default Environment Configurations
LOCAL_HOST_ENABLED = os.getenv("LOCAL_HOST_ENABLED", "true")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-q4_K_M")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "180"))
PROMPT_PATH = os.getenv("PROMPT_PATH", "prompt.txt")

# Logger
logger = logging.getLogger("fastapi_logs")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "NOTSET").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    filename=os.getenv("LOG_FILE", "server.log"),
    filemode="a", 
)

# Pydantic Models 
class ExtractedItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    price: float
    quantity: int = 1


class ExtractReceiptResponse(BaseModel):
    items: List[ExtractedItem]
    sst: float = 0  # percentage, e.g. 6% of subtotal
    serviceCharge: float = 0  # percentage, e.g. 10% of subtotal
    notes: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize PaddleOCR once at startup (heavy model load).
    """
    app.state.ocr = PaddleOCR(
        text_detection_model_name=os.getenv("PADDLE_DET_MODEL", "PP-OCRv5_server_det"),
        text_recognition_model_name=os.getenv("PADDLE_REC_MODEL", "PP-OCRv5_server_rec"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    yield

# --- FastAPI App & Middleware ---
# Limit to 2 requests per minute
limiter = RateLimiter(times=2, seconds=60)
app = FastAPI(
    title="ReceiptSplitter FastAPI",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _first_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of the first JSON object from a model response.
    Handles common cases where the model wraps JSON in markdown code fences (```json ... ```).
    """
    # remove ```json ... ``` fences if present
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```", "", text)

    # Find the first {...} block (naive but practical for MVP)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    candidate = text[start : end + 1]
    logger.info("JSON Output %s", candidate)
    return json.loads(candidate)


def _load_prompt_template() -> str:
    """
    Loads the prompt template from prompt.txt file.
    Raises ValueError if the file is missing, empty, or lacks the {{TEXT_DATA}} placeholder.
    """
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        logger.error("Prompt file not found: %s", PROMPT_PATH)
        raise ValueError(f"Prompt file not found: {PROMPT_PATH}")
    except Exception as e:
        logger.error("Failed to read prompt file %s: %s", PROMPT_PATH, e)
        raise ValueError(f"Failed to read prompt file {PROMPT_PATH}: {e}")

    if not template.strip():
        raise ValueError(f"Prompt file {PROMPT_PATH} is empty")

    if "{{TEXT_DATA}}" not in template:
        raise ValueError(f"Prompt file {PROMPT_PATH} must contain {{TEXT_DATA}} placeholder")

    return template


# Module-level initialization: load prompt template at startup
try:
    _PROMPT_TEMPLATE = _load_prompt_template()
except ValueError as e:
    logger.warning("Failed to load prompt template: %s. Requests will fail.", e)
    _PROMPT_TEMPLATE = ""


def _build_prompt_from_rec_texts(rec_texts: Sequence[str]) -> str:
    """
    Builds the prompt by replacing {{TEXT_DATA}} with the OCR recognized texts.
    """
    joined = "[\n" + ",\n".join([f'  {json.dumps(t)}' for t in rec_texts]) + "\n]"
    return _PROMPT_TEMPLATE.replace("{{TEXT_DATA}}", joined)


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes (cv2.imdecode returned None).")
    return img


def _extract_rec_texts_from_ocr_result(result: Any) -> List[str]:
    """
    Normalizes PaddleOCR outputs to a list of recognized text strings (rec_texts).

    Supports:
    [to be updated]
    """
    rec_texts: List[str] = []

    try:
        # Case 1: API Response ocrResults: [{'prunedResults:{...}'}]
        if isinstance(result, dict) and result:
            return result['ocrResults'][0]['prunedResult']['rec_texts']
        else: 
            # Case 2: PaddleOCR Local Output
            if isinstance(result, list):
                return result[0].get('rec_texts', [])
            
        raise Exception("Unrecognized PaddleOCR result format.")
    except Exception as e:
        return rec_texts


def _run_paddle_ocr(image_bytes: bytes) -> Tuple[List[str], Dict[str, Any]]:
    """
    Runs PaddleOCR for text detection and recognition.
    Returns (rec_texts, debug_info) tuple containing recognized text lines and debug metadata.
    """

    if LOCAL_HOST_ENABLED.lower() == "true":

        # Use Local PaddleOCR to Process Image
        img = _decode_image_bytes(image_bytes)

        # Prefer predict() if available (matches your test.py usage), else fallback to ocr()
        if hasattr(app.state.ocr, "predict"):
            try:
                result = app.state.ocr.predict(img)  # type: ignore[attr-defined]
                rec_texts = _extract_rec_texts_from_ocr_result(result)
                return rec_texts, {"mode": "predict", "raw_type": str(type(result)), "data": result}
            except Exception as e:
                result = app.state.ocr.ocr(img, cls=False)  # type: ignore[attr-defined]
                rec_texts = _extract_rec_texts_from_ocr_result(result)
                return rec_texts, {"mode": "ocr", "raw_type": str(type(result)), "data": result}
    else:
        # Use PaddleOCRv5 API Service (Baidu AI Studio)
        img = base64.b64encode(image_bytes).decode('ascii')

        headers = {
            "Authorization": f"token {os.getenv('PP_AI_STUDIO_TOKEN')}",
            "Content-Type": "application/json"
        }

        payload = {
            "file": img,
            "fileType": 1,  # For PDF Docs, set `fileType` to 0; for images, set to 1
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useTextlineOrientation": False,
        }

        response = requests.post(f"{os.getenv('PP_AI_STUDIO_URL')}", json=payload, headers=headers)
        result = response.json()['result']
        rec_texts = _extract_rec_texts_from_ocr_result(result)
        return rec_texts, {"mode": "api", "raw_type": str(type(result)), "data": json.dumps(result)}


async def _infer_with_ollama_from_text(rec_texts: Sequence[str]) -> Dict[str, Any]:
    """
    Calls Ollama's chat API (text-only) and requests STRICT JSON output.
    Logs the prompt, payload, and the model's JSON output.
    """
    prompt = _build_prompt_from_rec_texts(rec_texts)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_S) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    # Ollama typically returns:
    # {"message":{"role":"assistant","content":"..."} , ...}
    content = (
        (data.get("message") or {}).get("content")
        or data.get("response")
        or ""
    )
    if not content:
        raise ValueError("Empty model response content.")

    return _first_json_object(content)


def _coerce_response(model_json: Dict[str, Any]) -> ExtractReceiptResponse:
    """
    Coerce untrusted model output into a validated API response.
    Extracts items (name, price, quantity) and optional taxes/charges.
    Filters out non-item lines (e.g., tax, subtotal, payment info).
    """
    raw_items = model_json.get("items") or []
    items: List[ExtractedItem] = []

    for idx, it in enumerate(raw_items):
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        if not name:
            continue

        # Defensive: filter out obvious summary/tax/surcharge/payment lines if the model still includes them.
        bad = name.lower()
        if any(
            k in bad
            for k in [
                "tax",
                "sst",
                "service tax",
                "gst",
                "vat",
                "service charge",
                "round",
                "rounding",
                "subtotal",
                "total",
                "grand total",
                "nett",
                "taxable",
                "discount",
                "change",
                "cash",
                "card",
                "qr",
                "payment",
                "amount due",
            ]
        ):
            continue
        try:
            price = float(it.get("price"))
        except Exception:
            continue
        try:
            quantity = int(it.get("quantity") or 1)
        except Exception:
            quantity = 1
        if quantity <= 0:
            quantity = 1

        items.append(ExtractedItem(name=name, price=price, quantity=quantity))

    # Optional percentages
    try:
        sst = float(model_json.get("sst") or 0)
    except Exception:
        sst = 0.0
    try:
        service = float(model_json.get("serviceCharge") or 0)
    except Exception:
        service = 0.0

    if LOCAL_HOST_ENABLED.lower() == "true":
        return ExtractReceiptResponse(
            items=items,
            sst=sst,
            serviceCharge=service,
            notes=f"Ollama inference via {OLLAMA_BASE_URL} model={OLLAMA_MODEL}.")
    else:
        return ExtractReceiptResponse(
            items=items,
            sst=sst,
            serviceCharge=service,
            notes=f"Gemini Cloud inference via API key.")

@app.post("/extract", response_model=ExtractReceiptResponse, dependencies=[Depends(limiter)])
async def extract_receipt(
    image: UploadFile = File(...)
) -> ExtractReceiptResponse:
    """
    Receives an image upload and returns extracted receipt info.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    logger.info("Received /extract upload filename=%s content_type=%s size_bytes=%d", image.filename, image.content_type, len(image_bytes))
    
    # 1) PaddleOCR: detect + recognize text
    try:
        rec_texts, ocr_debug = _run_paddle_ocr(image_bytes)
    except Exception as e:
        logger.exception("OCR failed: %s", e)
        raise HTTPException(status_code=502, detail=f"OCR failed: {e}")

    if not rec_texts:
        logger.warning("OCR produced no text. debug=%s", ocr_debug)
        raise HTTPException(status_code=422, detail="OCR produced no recognized text.")
    else:  
        rec_texts = [re.sub(r'[^\x00-\x7F]+', '', t) for t in rec_texts]

    logger.info("OCR ok: lines=%d debug=%s", len(rec_texts), ocr_debug)
    logger.debug("OCR sample: %s", rec_texts[:20])
    
    # 2) Ollama/Gemini Gemma3: parse OCR text to strict JSON
    # Local / Gemini Cloud Inference
    if LOCAL_HOST_ENABLED.lower() == "true":
        try:
            model_json = await _infer_with_ollama_from_text(rec_texts)
        except httpx.HTTPError as e:
            logger.exception("Ollama HTTP error: %s", e)
            raise HTTPException(
                status_code=502,
                detail=f"Ollama call failed ({type(e).__name__}). Is Ollama running at {OLLAMA_BASE_URL}?",
            )
        except Exception as e:
            logger.exception("Model inference/parsing failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Model inference/parsing failed: {e}")

    else:
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

            response = client.models.generate_content(
                model=os.getenv("CLOUD_MODEL", "gemma-3-4b-it"),
                contents=_build_prompt_from_rec_texts(rec_texts)
            )
            model_json = _first_json_object(response.text)
            resp = print(_coerce_response(model_json))
        except Exception as e:
            logger.exception("Cloud model inference/parsing failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Cloud model inference/parsing failed: {e}")
    
    resp = _coerce_response(model_json)
    logger.info("Extraction ok: items=%d sst=%s serviceCharge=%s", len(resp.items), resp.sst, resp.serviceCharge)
    return resp