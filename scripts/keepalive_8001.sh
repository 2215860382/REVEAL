#!/usr/bin/env bash
while true; do
  env -u http_proxy -u https_proxy NO_PROXY='*' curl -s -m 20 http://127.0.0.1:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' >/dev/null 2>&1
  sleep 180
done
