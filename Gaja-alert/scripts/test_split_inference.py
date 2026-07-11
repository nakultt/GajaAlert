"""Smoke test and verification for the hybrid NPU + GPU local server.

Usage:
    C:\\Users\\qcwor\\llm\\geniex-env\\Scripts\\python.exe scripts\\test_split_inference.py

Requires the server to be running:
    C:\\Users\\qcwor\\llm\\geniex-env\\Scripts\\python.exe scripts\\serve_e2b_split.py
"""

import json
import urllib.request
import time

BASE = "http://127.0.0.1:8082"


def main():
    print("--- 1. Verification: Running Health Check ---")
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
            health = json.loads(r.read().decode())
            print("Response:", json.dumps(health, indent=2))
            assert health["status"] == "ok"
            assert "HTP0,GPUOpenCL" in health["device"]
            print("Verification passed! Server is active on hybrid NPU + GPU.")
    except Exception as e:
        print("Health check failed. Make sure the serve_e2b_split.py server is running!")
        print("Error:", str(e))
        return

    print("\n--- 2. Running Inference Test ---")
    messages = [
        {"role": "user", "content": "Explain quantum computing in exactly one short sentence."}
    ]
    body = json.dumps({
        "messages": messages, 
        "temperature": 0.1, 
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False}
    }).encode()
    
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read().decode())
        dt = time.time() - t0
        print(f"Response received in {dt:.2f}s:")
        print(json.dumps(out, indent=2))
        print("\nGenerated Text:")
        print(out["choices"][0]["message"]["content"])
    except Exception as e:
        print("Inference call failed:", str(e))


if __name__ == "__main__":
    main()
