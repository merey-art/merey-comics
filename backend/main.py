#!/usr/bin/env python3
import tempfile
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import cv2
import httpx
from bs4 import BeautifulSoup
from curl_cffi import requests as impersonate_requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from panel_detection_cv2 import (
    build_frontend_response,
    build_panel_json,
    detect_panel_boxes,
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 15

# Cloudflare's WAF fingerprints the TLS/HTTP stack: plain `requests`
# negotiates HTTP/1.1 and gets a 403, while a real browser (and curl, by
# default) speaks HTTP/2 and passes. httpx with http2=True mimics that.
http_client = httpx.Client(http2=True, timeout=REQUEST_TIMEOUT, follow_redirects=True)

app = FastAPI(title="Guided View API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessPageRequest(BaseModel):
    image_url: HttpUrl


class ParseChapterRequest(BaseModel):
    chapter_url: HttpUrl


def guess_referer(url: str) -> str:
    """CDN hosts (e.g. img.batcave.biz) hotlink-protect on Referer and
    return a Cloudflare WAF 403 for requests missing it. Reconstruct the
    parent site's root as a same-site referer to pass that check."""
    parsed = urlparse(url)
    labels = parsed.netloc.split(".")
    root_domain = ".".join(labels[-2:]) if len(labels) >= 2 else parsed.netloc
    return f"{parsed.scheme}://{root_domain}/"


IMAGE_CACHE_MAX_ENTRIES = 100
image_cache: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()


def download_image(url: str) -> tuple[bytes, str]:
    """Fetch image bytes + content-type, going through our own cache so a
    page processed via /api/process-page doesn't need a second round-trip
    to the origin when the browser then requests it via /api/image-proxy."""
    cached = image_cache.get(url)
    if cached is not None:
        image_cache.move_to_end(url)
        return cached

    headers = {**REQUEST_HEADERS, "Referer": guess_referer(url)}
    try:
        resp = http_client.get(url, headers=headers)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download image: {exc}") from exc

    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    result = (resp.content, content_type)

    image_cache[url] = result
    image_cache.move_to_end(url)
    if len(image_cache) > IMAGE_CACHE_MAX_ENTRIES:
        image_cache.popitem(last=False)

    return result


def download_image_bytes(url: str) -> bytes:
    content, _ = download_image(url)
    return content


def build_proxy_url(image_url: str) -> str:
    return f"/api/image-proxy?url={quote(image_url, safe='')}"


def detect_panels_from_bytes(image_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        image = cv2.imread(tmp.name)

    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    height, width = image.shape[:2]
    ordered_boxes = detect_panel_boxes(image)

    return build_panel_json(ordered_boxes, (height, width))


@app.post("/api/process-page")
def process_page(payload: ProcessPageRequest) -> dict:
    image_url = str(payload.image_url)
    image_bytes = download_image_bytes(image_url)
    panel_json = detect_panels_from_bytes(image_bytes)
    frontend_json = build_frontend_response(panel_json)
    frontend_json["source_url"] = image_url
    # The browser can't hotlink CDN images directly (they require a
    # same-site Referer the client can't spoof), so hand back a same-origin
    # proxy URL instead of the raw CDN URL.
    frontend_json["image_url"] = build_proxy_url(image_url)
    return frontend_json


@app.get("/api/image-proxy")
def image_proxy(url: str = Query(...)) -> Response:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid url")

    content, content_type = download_image(url)
    return Response(content=content, media_type=content_type, headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/parse-chapter")
def parse_chapter(payload: ParseChapterRequest) -> dict:
    chapter_url = str(payload.chapter_url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": guess_referer(chapter_url),
    }
    try:
        resp = impersonate_requests.get(
            chapter_url, headers=headers, impersonate="chrome120", timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
    except impersonate_requests.RequestsError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch chapter page: {exc}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    page_urls: list[str] = []

    for img in soup.select(".reader__item_wrap img"):
        src = img.get("data-src") or img.get("data-original") or img.get("src")
        if src:
            page_urls.append(urljoin(chapter_url, src))

    if not page_urls:
        for img in soup.select("img[data-src]"):
            src = img.get("data-src")
            if src:
                page_urls.append(urljoin(chapter_url, src))

    if not page_urls:
        raise HTTPException(status_code=404, detail="No pages found in chapter HTML")

    return {"chapter_url": chapter_url, "page_count": len(page_urls), "pages": page_urls}


static_dir = Path(__file__).resolve().parent.parent / "frontend"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
