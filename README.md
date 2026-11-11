# Maya — Local Voice Agent

Maya is a local-first, real-time voice agent built with LiveKit Agents. It combines streaming Moonshine speech recognition, a local llama.cpp-compatible LLM, Piper CPU text-to-speech, optional Kokoro ONNX comparison, and PostgreSQL vector retrieval. The project focuses on the part that makes voice agents feel conversational: fast end-of-turn response and stable audio delivery.

![Maya frontend preview](docs/frontend-preview.svg)

> **Local development demo — not production authentication.** The demo creates guest LiveKit tokens and allows a local browser to start calls. Before exposing it publicly, add real user authentication, authorization, HTTPS, rate limits, secret management, and production CORS policy.

## What this project demonstrates

- Custom LiveKit STT and TTS adapters with asynchronous streams.
- Process-level model prewarming so the first call does not pay every model-load cost.
- Agent readiness gating: the call API dispatches the agent and waits for its voice_agent.ready=true marker before returning the user's token.
- Sentence-aware LLM chunking so speech can begin before the full answer is complete.
- CPU Piper TTS to avoid competing with the local LLM on Apple Metal.
- Optional RAG with local embeddings and PostgreSQL/pgvector.
- Turn-level latency events for STT, endpointing, RAG, LLM, TTS, cancellations, interruptions, and real-time factor (RTF).
- A React frontend with readiness progress, live captions, microphone controls, and audio-unlock handling.

## Architecture

~~~mermaid
flowchart LR
    B[React browser] -->|POST /api/calls/start| API[FastAPI API]
    API -->|create room + dispatch| LK[LiveKit]
    API -->|poll voice_agent.ready| LK
    API -->|return token after ready| B
    B <-->|WebRTC audio + captions| LK
    LK --> AG[LiveKit voice agent]
    AG --> PRE[Prewarmed resources]
    PRE --> STT[Moonshine streaming STT]
    PRE --> VAD[Silero VAD + turn detector]
    PRE --> TTS[Piper CPU TTS]
    AG --> LLM[llama.cpp OpenAI-compatible server]
    AG --> RAG[Async RAG retrieval]
    RAG --> PG[(PostgreSQL + pgvector)]
    LLM --> TTS
    TTS --> LK
~~~

### Request path

1. The browser calls POST /api/calls/start.
2. FastAPI creates a room and dispatches the LiveKit agent.
3. The worker prewarms or reuses STT, VAD, RAG, LLM, and TTS resources.
4. The agent publishes voice_agent.ready=true.
5. FastAPI returns the user token only after readiness is observed.
6. The browser joins, enables its microphone, and the agent greets the user.

## Model choices and trade-offs

The design goal was a completely local voice agent: audio, transcription, retrieval, language generation, and speech synthesis should run on the developer's machine. That gave us privacy and repeatable experiments, but it also meant that every model had to share a laptop's CPU, memory, and Apple Metal resources. We chose models by measuring the end-to-end experience, not by choosing the largest model available.

| Component | Selected model/backend | Why we chose it | Problem it solved / trade-off |
| --- | --- | --- | --- |
| LLM | Qwen2.5-3B-Instruct, quantized as GGUF and served by llama.cpp | A mid-sized 3.09B instruction model gave us a useful quality/latency balance for short spoken answers. It was large enough to answer engineering and support questions well, but small enough to run locally with low observed time-to-first-token. | The first MLX serving path was fast in isolation, but our overlapping agent workload exposed unreliable turn handling and latency tails. This is a limitation we observed in our configuration, not a claim that MLX can never batch requests. llama.cpp gave us an explicit local HTTP server with streaming chat completions, parallel decoding, and continuous-batching support. The trade-off is that model quality and speed still depend heavily on quantization, context size, and Metal/CPU settings. |
| Initial LLM path | MLX / `mlx-lm` | It was the natural first choice on Apple Silicon because it uses the MLX stack and Metal efficiently. | Under agent turns, preemptive generations, and other local accelerator work, our original path did not give us the concurrency behavior and latency predictability we needed. We moved to llama.cpp for a clearer serving boundary and more explicit multi-request controls. |
| STT baseline | Local Whisper | Whisper was accurate and familiar, so it was an obvious first experiment. | The official `transcribe()` path reads an audio file and processes it in sliding 30-second windows. Our LiveKit agent needed incremental audio, interim transcripts, and quick finalization while the user was still speaking. Whisper is streamable through other wrappers, but the baseline we tried did not provide the truly streaming interface this project needed on the Mac. |
| STT | Moonshine `medium-streaming` | Moonshine was designed for on-device streaming, exposes interim results, and had the right combination of accuracy, speed, and model size for this laptop. | It removed the need to wait for a complete recording and allowed endpointing to start from partial transcripts. In the measured turns, normal `transcript_s` was about 0.322–0.462 s; one 2.251 s transcript outlier remains visible in the local log history. |
| VAD | Silero VAD on CPU | Detects speech and silence locally without competing with the LLM or TTS accelerator path. | CPU VAD reduced accelerator contention, although the logs showed that VAD can still become slower than real time when the whole worker is overloaded. |
| Turn detection | LiveKit v1-mini detector | Predicts when the user has finished speaking instead of waiting for a long fixed silence. | It improved responsiveness, but endpointing is a product trade-off: too short interrupts users, too long feels slow. Normal observed endpointing was roughly 0.400–0.651 s in the Piper-era sample. |
| Embeddings tried first | `sentence-transformers/all-mpnet-base-v2` | Strong general-purpose semantic-search quality made it a reasonable first RAG model. | It is a much larger 768-dimensional model than the final choice, and its cold load plus CPU query work was too heavy for a latency-sensitive voice path on this machine. The retained project history records several-second cold embedding startup and roughly 600–684 MB RAG-ready memory; we do not publish a made-up MPNet per-query number because an apples-to-apples run was not retained. |
| Embeddings | `Snowflake/snowflake-arctic-embed-xs` on CPU | Snowflake's XS model has about 22M parameters and 384-dimensional vectors, so it was small enough to keep local and fast enough for interactive retrieval while still offering useful retrieval quality. | It replaced the heavier MPNet path. In the later local logs, warm query embedding was commonly 0.007–0.263 s, with a 0.027 s median in the sampled Piper-era turns; total warm RAG was commonly 0.012–0.325 s, with a 0.051 s median. The first RAG prewarm still took about 8.46 s, which is why it happens before the user joins. |
| Default TTS | Piper `en_US-lessac-low` through ONNX Runtime on CPU | Piper is a fast, local neural TTS system, and CPU inference keeps speech synthesis away from the LLM's Metal workload. | It removed the main Kokoro MLX contention problem. In representative short-turn logs, Piper first PCM was about 0.023–0.329 s and RTF stayed well below 1.0; the trade-off is less expressive prosody than Kokoro and a greater risk of a mechanical sound if text is split into overly small chunks. |
| Comparison TTS | Kokoro ONNX | We kept a CPU/ONNX Kokoro adapter so voice quality can be compared without putting Kokoro on the same MLX/Metal path as the LLM. | Kokoro can sound more expressive, but it has a larger first-audio and resource budget in this project. The multi-second numbers below refer specifically to the historical Kokoro MLX path, not to a controlled benchmark of the current Kokoro ONNX adapter. |
| Vector store | PostgreSQL + pgvector | A local persistent store makes document ingestion and semantic search easy to reproduce. | It adds a database boundary, so retrieval is measured as embedding time plus database time and is skipped for greetings or short casual turns. |

Model weights are intentionally not committed. Download or place the assets locally, then point the environment variables at them.

Capability references checked while writing this section: [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [`mlx-lm` server documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md), [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [OpenAI Whisper README](https://github.com/openai/whisper/blob/main/README.md), [Moonshine](https://github.com/moonshine-ai/moonshine), [Snowflake Arctic Embed XS model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs), [Sentence Transformers model comparison](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html), [Piper](https://github.com/rhasspy/piper), [Piper streaming notes](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/STREAMING.md), and [Kokoro ONNX model releases](https://github.com/thewh1teagle/kokoro-onnx/releases).

## Latency story

For a voice agent, end-of-turn latency matters more than text-only response latency: the user experiences the silence between stopping speech and hearing the first spoken response. We therefore measure every boundary separately: transcript finalization, endpointing, RAG embedding/database time, LLM time-to-first-token, TTS time-to-first-PCM, TTS real-time factor, and first published audio.

### Measured local metrics

The following numbers are observations from this Mac's development logs, not a controlled cross-machine benchmark. They are useful for explaining the engineering decisions, but they are not guarantees for another machine or model quantization.

| Stage | Local observation | What it told us |
| --- | ---: | --- |
| Moonshine prewarm | About 1.23–1.26 s when the STT model was loaded | Cold startup belongs in worker readiness, not in the user's first turn. |
| Moonshine turn transcription | 0.322–0.462 s in normal Piper-era turns; 2.251 s outlier | Streaming STT was no longer the dominant steady-state bottleneck, but CPU contention can still create tails. |
| Endpointing | About 0.400–0.651 s in normal turns | Silence policy is part of perceived latency and must be tuned against interruption quality. |
| Snowflake query embedding | 0.007–0.263 s; 0.027 s median in the sampled Piper-era turns | The smaller CPU embedding model made warm RAG practical for voice. |
| Warm RAG total | 0.012–0.325 s; 0.051 s median in the sampled Piper-era turns | RAG was usually not the source of the multi-second TTS spikes. |
| Qwen/llama.cpp TTFT | Roughly 0.10–0.44 s in the sampled later runs | A small local model plus short prompts could produce usable text quickly. |
| Piper first PCM | Roughly 0.023–0.329 s for representative short chunks | CPU TTS stayed ahead of playback and avoided Metal contention. |
| Historical Kokoro MLX first PCM | About 1.0–6.7 s, with RTF up to 2.52 | MLX/Metal contention, repeated synthesis work, and non-streaming behavior created the largest tail. |
| First published audio | Piper-era turns were commonly around 0.63–1.26 s; historical Kokoro MLX spikes reached about 7.27 s | The end-to-end metric is what the caller feels, so optimizing only LLM TTFT was insufficient. |

`transcript_s` is a turn-level timing from the agent trace, not a claim that the neural network itself takes exactly that long. Likewise, `first_audio_e2e` is an observed/additive diagnostic estimate assembled from the pipeline timestamps; hardware, sentence length, model size, queueing, and concurrent work can change it substantially.

| Metric | Piper CPU | Kokoro MLX observed locally |
| --- | ---: | ---: |
| TTS first PCM | 0.023–0.329 s in representative short chunks | 1.0–6.7 s |
| TTS real-time factor | Usually well below 1.0; sampled values 0.030–0.189 | 0.28–2.52 |
| Wall-clock first audio | Commonly about 0.63–1.26 s in later Piper-era turns | Up to 7.27 s in observed spikes |
| Main trade-off | Faster, less expressive voice | More expressive, but sensitive to accelerator contention |

### What caused the latency and what fixed it

The original shape of the bottleneck was:

~~~text
user stops speaking
  -> final STT + endpointing
  -> local LLM produces text
  -> Kokoro MLX waits for shared Metal resources
  -> phonemization/synthesis completes
  -> audio is published
~~~

Kokoro MLX and the local LLM both used Apple Metal. The LLM could produce its first token quickly, while TTS waited for accelerator time, sometimes generated speech slower than real time, and received several small text chunks. The logs also showed occasional event-loop stalls during synchronous initialization or inference, including roughly 174–250 ms warnings. That combination made the voice feel late and sometimes fragmented.

The main fixes were:

- prewarm STT, VAD, turn detection, embeddings, LLM connectivity, and TTS before readiness is reported;
- move the default TTS to Piper ONNX on CPU so the LLM keeps the accelerator;
- run synchronous model work in worker threads and return audio through asynchronous streams instead of blocking the LiveKit event loop;
- start speech from a short complete sentence, but avoid endlessly splitting an answer into tiny TTS requests;
- run RAG asynchronously with a timeout, skip it for chitchat, cache the embedding/vector-store objects, and bound the retrieved context;
- avoid speculative generation when RAG can still change the prompt, because the first generation would otherwise be discarded and repeated;
- record queue wait, first PCM, RTF, cancellation, interruption, and wall-clock first-audio events separately.

The remaining quality trade-off is conversationality. Piper is fast and stable, but it is less expressive than Kokoro. Prompt instructions can improve phrasing, contractions, sentence rhythm, and pauses, but they cannot create richer vocal prosody. Very small TTS chunks can also sound mechanical even when their first-audio metric is excellent. The next quality experiment should use a sentence/clauses buffer and compare larger, smoother Piper chunks with Kokoro ONNX under the same latency budget.

### Making the voice conversational

The agent instructions now ask Maya to use contractions, vary sentence length and openings, acknowledge the user briefly when useful, and avoid formal written prose. This improves the language rhythm, but it cannot create prosody by itself. Piper is intentionally the fast default and is less expressive than Kokoro, while very small streaming chunks can still make any voice sound clipped or robotic. A future quality-focused experiment would keep the same sentence buffer but synthesize larger clauses or complete sentences, then compare a more expressive voice model against the latency budget.

Run the local Piper benchmark with llama-server already running:

~~~sh
.venv/bin/python scripts/bench_piper.py --repeats 5
~~~

Look for first_audio_p50_s < 1.0 and rtf_p50 < 1.0 on the machine being tested. Use the sanitized example at examples/sample-latency.log as a guide to the events emitted by the agent.

## Quick start

### 1. Install dependencies

Python 3.10–3.14 and Node.js are supported. Python 3.12 is the recommended development version.

~~~sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cp .env.example .env
cd frontend
npm ci
cd ..
~~~

### 2. Start PostgreSQL/pgvector

~~~sh
docker compose up -d pgvector
~~~

Start a local LiveKit server separately and use the matching development API key, secret, and WebSocket URL in .env.

### 3. Prepare local models

Download a Piper voice into models/piper:

~~~sh
.venv/bin/python -m pip install piper-tts
.venv/bin/python -m piper.download_voices \
  --download-dir models/piper en_US-lessac-low
~~~

Place Moonshine model assets in models/moonshine as required by the installed moonshine-voice package. Set MOONSHINE_MODEL_DIR and MOONSHINE_MODEL_ARCH=medium-streaming in .env.

For the optional Kokoro ONNX comparison backend, place its model and voice pack in models/kokoro, then run:

~~~sh
VOICE_TTS_BACKEND=kokoro-onnx .venv/bin/python agent.py start
~~~

Start a local llama.cpp server with a compatible GGUF model:

~~~sh
llama-server -m /path/to/model.gguf -ngl 99 -fa on -c 4096 \
  --parallel 1 --cache-prompt --host 127.0.0.1 --port 8080
~~~

### 4. Ingest your own documents

The repository does not include sample PDFs. Bring your own document and run:

~~~sh
.venv/bin/python ingest.py /path/to/your-document.pdf
~~~

The ingestor sanitizes NUL characters before inserting PDF text into PostgreSQL.

### 5. Run the services

In separate terminals:

~~~sh
# API
.venv/bin/uvicorn main:app --reload --port 8000

# LiveKit agent
.venv/bin/python agent.py start

# Frontend
cd frontend && npm run dev
~~~

Open the Vite URL shown in the frontend terminal. Verify the API first with:

~~~sh
curl http://localhost:8000/health
~~~

Expected response:

~~~json
{"status":"ok"}
~~~

## Configuration

Copy .env.example to .env. The most important settings are:

| Variable | Purpose |
| --- | --- |
| VOICE_TTS_BACKEND | piper by default; kokoro-onnx for comparison |
| PIPER_MODEL_PATH | Local Piper ONNX voice path |
| LOCAL_LLM_BASE_URL | llama.cpp OpenAI-compatible endpoint |
| RAG_ENABLED | Enables document retrieval |
| VOICE_AGENT_READY_TIMEOUT_S | Maximum wait before the API returns a readiness error |
| VOICE_FIRST_CHUNK_MAX_CHARS | Bounds the first spoken response chunk |

## Tests and checks

~~~sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -q
cd frontend && npm run lint && npm run build
~~~

GitHub Actions runs the same Python and frontend checks on every push and pull request.

## Project structure

~~~text
agent.py                    LiveKit worker, prewarm, orchestration, latency metrics
moonshine_streaming_stt.py  Custom streaming STT adapter
piper_tts.py                Custom Piper LiveKit TTS adapter
kokoro_onnx_tts.py          Optional Kokoro ONNX LiveKit TTS adapter
local_llm_client.py         Async OpenAI-compatible local LLM client
rag.py                      Local embeddings and PostgreSQL/pgvector retrieval
services/                   Call creation, dispatch, and readiness handshake
frontend/                   React voice-call interface
tests/                      Unit tests for adapters and latency behavior
examples/                   Sanitized sample observability output
~~~

## License and model terms

The application code and every model/voice package have separate licensing terms. Review the licenses of Piper, the selected voice, Moonshine, Kokoro, the LLM GGUF, embedding model, and any user-provided documents before redistributing a complete bundle. Add a project license before accepting external contributions.

