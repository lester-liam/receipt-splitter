"""
FastAPI draft endpoint for receipt extraction using Ollama `gemma3:4b-it-q4_K_M`.

Goal:
- Receive an image as bytes
- Run OCR (PaddleOCR) to detect/recognize text
- Send OCR text to Ollama (text model) for structured parsing
- Extract:
  - Mandatory: item name, price, quantity
  - Optional: SST %, service charge % (default 0)
- Return JSON matching the mock extraction shape used in `index.html`:

{
  "items": [{"id": "...", "name": "...", "price": 0.0, "quantity": 1}, ...],
  "sst": 0,
  "serviceCharge": 0,
}

Notes:
- This is intentionally "MVP draft": it includes a robust interface + placeholders.
- Parsing receipt text into structured items is non-trivial; we ask the model to output strict JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field

# Load Environment
load_dotenv()
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

app = FastAPI(
    title="Receipt Splitter MVP API",
    version="0.1.0",
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
    - PaddleOCR.ocr(img, ...) legacy output: list[list[ [box], (text, score) ]]
    - PaddleOCR.predict(img) newer output objects/dicts that expose rec_texts
    """
    rec_texts: List[str] = []

    # Case 1: object(s) with attribute `rec_texts`
    if isinstance(result, list) and result and hasattr(result[0], "rec_texts"):
        for res in result:
            rec_texts.extend([str(t) for t in getattr(res, "rec_texts", []) if str(t).strip()])
        return rec_texts

    # Case 2: dict(s) with key `rec_texts`
    if isinstance(result, list) and result and isinstance(result[0], dict) and "rec_texts" in result[0]:
        for res in result:
            rec_texts.extend([str(t) for t in (res.get("rec_texts") or []) if str(t).strip()])
        return rec_texts

    # Case 3: legacy `ocr` structure: [[ [box], (text, score) ], ...]
    if isinstance(result, list):
        # Sometimes it's wrapped as [page0] where page0 is list of lines
        page = result[0] if (len(result) == 1 and isinstance(result[0], list)) else result
        for line in page:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            text_part = line[1]
            if isinstance(text_part, (list, tuple)) and text_part:
                text = str(text_part[0]).strip()
                if text:
                    rec_texts.append(text)
        return rec_texts

    return rec_texts


def _run_paddle_ocr(image_bytes: bytes) -> Tuple[List[str], Dict[str, Any]]:
    """
    Runs PaddleOCR for text detection and recognition.
    Returns (rec_texts, debug_info) tuple containing recognized text lines and debug metadata.
    """
    img = _decode_image_bytes(image_bytes)

    # Prefer predict() if available (matches your test.py usage), else fallback to ocr()
    if hasattr(app.state.ocr, "predict"):
        result = app.state.ocr.predict(img)  # type: ignore[attr-defined]
        rec_texts = _extract_rec_texts_from_ocr_result(result)
        return rec_texts, {"mode": "predict", "raw_type": str(type(result))}

    result = app.state.ocr.ocr(img, cls=False)  # type: ignore[attr-defined]
    rec_texts = _extract_rec_texts_from_ocr_result(result)
    return rec_texts, {"mode": "ocr", "raw_type": str(type(result))}


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

    return ExtractReceiptResponse(
        items=items,
        sst=sst,
        serviceCharge=service,
        notes=f"Ollama inference via {OLLAMA_BASE_URL} model={OLLAMA_MODEL}.",
    )


@app.post("/extract", response_model=ExtractReceiptResponse)
async def extract_receipt(image: UploadFile = File(...)) -> ExtractReceiptResponse:
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

    # 2) Ollama Gemma3: parse OCR text to strict JSON
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

    resp = _coerce_response(model_json)
    logger.info("Extraction ok: items=%d sst=%s serviceCharge=%s", len(resp.items), resp.sst, resp.serviceCharge)
    return resp