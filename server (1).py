import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from model import TomatinhoGPT

CHECKPOINT = Path("artifacts/tomatinho.pt")

# CORS: lista separada por vírgula. Vazia = nenhuma origem externa permitida.
# Exemplo: ALLOWED_ORIGINS="https://seu-usuario.hf.space,https://tomatinho.exemplo.com"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# Limite de requisições por IP a /generate (janela de 60 segundos).
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
RATE_WINDOW_SECONDS = 60
# Em Hugging Face Spaces a conexão passa por proxy; o IP real vem em X-Forwarded-For.
TRUST_PROXY = os.environ.get("TRUST_PROXY", "1") == "1"

app = FastAPI(title="Tomatinho AI API", version="0.1.1")
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"]
)

model = None
stoi = {}
itos = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "desconhecido"


def check_rate_limit(ip: str) -> int | None:
    """Retorna None se permitido, ou segundos até liberar se bloqueado."""
    now = time.monotonic()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > RATE_WINDOW_SECONDS:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return max(1, int(RATE_WINDOW_SECONDS - (now - q[0])) + 1)
        q.append(now)
        return None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=120, ge=1, le=300)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)


@app.on_event("startup")
def load_model():
    global model, stoi, itos
    if not CHECKPOINT.exists():
        return
    ckpt = torch.load(CHECKPOINT, map_location=device)
    vocab = ckpt["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    model = TomatinhoGPT(len(vocab), **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()


@app.get("/")
def health():
    return {"name": "Tomatinho AI", "version": "0.1.1",
            "model_loaded": model is not None, "device": device}


@app.post("/generate")
def generate(req: GenerateRequest, request: Request):
    retry = check_rate_limit(client_ip(request))
    if retry is not None:
        return JSONResponse(
            status_code=429,
            content={"detail": "Muitas requisições. Tente novamente em instantes."},
            headers={"Retry-After": str(retry)},
        )
    if model is None:
        raise HTTPException(503, "Modelo não treinado. Execute train.py primeiro.")
    encoded = [stoi[c] for c in req.prompt if c in stoi]
    if not encoded:
        raise HTTPException(400, "O prompt não contém caracteres conhecidos pelo modelo.")
    x = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(x, req.max_new_tokens, req.temperature)
    ids = y[0].tolist()
    result = "".join(itos[int(i)] for i in ids)
    continuation = "".join(itos[int(i)] for i in ids[len(encoded):])
    return {"text": result, "continuation": continuation,
            "note": "Protótipo experimental; pode gerar texto incoerente."}


# Interface servida pelo próprio servidor (mesma origem: sem CORS e sem domínio extra).
# Acesso: https://SEU-SPACE.hf.space/ui/
WEB_DIR = Path("web")
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")

