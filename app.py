import os
import asyncio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncAzureOpenAI

# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────
load_dotenv()
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
API_KEY  = os.getenv("AZURE_OPENAI_KEY", "")
API_VERSION = "2024-12-01-preview"
DEPLOYMENT  = "model-router"            # adapt if you changed the name

if not (ENDPOINT and API_KEY):
    raise RuntimeError(
        "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY in environment or .env")

# ──────────────────────────────────────────────────────────────
# FastAPI setup
# ──────────────────────────────────────────────────────────────
app = FastAPI()
with open("index.html", encoding="utf-8") as fp:
    HTML_PAGE = fp.read()

@app.get("/", include_in_schema=False)
async def root():
    return HTMLResponse(HTML_PAGE)

@app.websocket("/ws")
async def chat(ws: WebSocket):
    await ws.accept()
    history = [{"role": "system", "content": "You are a helpful assistant."}]

    # Use the SDK as an async context-manager so connections close cleanly
    async with AsyncAzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
    ) as client:

        try:
            while True:
                user_msg = await ws.receive_text()
                history.append({"role": "user", "content": user_msg})

                stream = await client.chat.completions.create(
                    model=DEPLOYMENT,
                    messages=history,
                    stream=True,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=1024,
                )

                full_reply = ""
                model_used  = "unknown"

                async for chunk in stream:
                    # ── ① Skip heartbeat / empty-choice events ──────────────────
                    if not getattr(chunk, "choices", []):
                        continue

                    choice = chunk.choices[0]
                    delta  = choice.delta

                    # stream tokens to browser
                    token = delta.content or ""
                    if token:
                        await ws.send_text(token)
                        full_reply += token

                    # capture model once (same for all chunks)
                    if model_used == "unknown":
                        model_used = chunk.model

                    # graceful end of answer
                    if choice.finish_reason is not None:
                        break

                # add assistant reply to history
                history.append({"role": "assistant", "content": full_reply})

                # tell the UI which model was selected
                await ws.send_text(f"<<MODEL::{model_used}>>")

        except WebSocketDisconnect:
            # browser closed – just exit the coroutine
            pass
        except asyncio.CancelledError:
            # server shutting down (reloader etc.)
            raise
        finally:
            await ws.close()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
