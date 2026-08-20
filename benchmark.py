import asyncio
import time
import httpx
import statistics

COORDINATOR = "http://localhost:8000"

PROMPTS = [
    ("short",  "What is 2 + 2?",                          20),
    ("medium", "Explain how neural networks work.",         80),
    ("long",   "Write a detailed explanation of the history of the internet.", 150),
]

async def benchmark():
    results = {}
    async with httpx.AsyncClient(timeout=300) as client:
        for label, prompt, max_tok in PROMPTS:
            times = []
            for _ in range(3):  # 3 runs per prompt
                t0 = time.time()
                resp = await client.post(f"{COORDINATOR}/infer", json={
                    "prompt": prompt,
                    "max_new_tokens": max_tok
                })
                elapsed = time.time() - t0
                data = resp.json()
                tokens = len(data["response"].split())
                times.append(elapsed)
                print(f"[{label}] {elapsed:.2f}s | ~{tokens} tokens | {tokens/elapsed:.1f} tok/s")
            results[label] = {
                "avg_s": statistics.mean(times),
                "min_s": min(times),
                "max_s": max(times),
            }
    
    print("\n=== Summary ===")
    for label, r in results.items():
        print(f"{label:8} avg={r['avg_s']:.2f}s  min={r['min_s']:.2f}s  max={r['max_s']:.2f}s")

asyncio.run(benchmark())