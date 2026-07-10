#!/usr/bin/env bash
# Keep remote vLLM endpoints warm during a long build/eval so their 5-min
# idle-self-kill doesn't fire during the pipeline's local-only phases
# (frame extraction / SigLIP grouping can run >5 min with no LLM call).
# Sends a tiny 1-token chat completion to each endpoint every INTERVAL sec.
# Start alongside the build; STOP it when the experiment is done so the
# endpoints' own idle-kill can then reap them.
#
#   setsid nohup bash scripts/_vllm_keepalive.sh > outputs/logs/vllm_keepalive.log 2>&1 &
set -u
ENDPOINTS=${ENDPOINTS:-http://10.82.1.145:8002,http://10.82.1.145:8005}
MODEL=${MODEL:-Qwen3.5-27B}
INTERVAL=${INTERVAL:-60}          # seconds; must be < the endpoints' idle timeout (300)
export NO_PROXY=10.82.1.145,127.0.0.1,localhost no_proxy=$NO_PROXY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
IFS=',' read -ra EPS <<< "$ENDPOINTS"
echo "[keepalive] pinging ${#EPS[@]} endpoint(s) every ${INTERVAL}s → keeping vLLM warm"
while :; do
  for ep in "${EPS[@]}"; do
    code=$(curl -s -o /dev/null --max-time 5 --noproxy '*' -w '%{http_code}' \
      "${ep%/}/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}" 2>/dev/null)
    echo "[$(date '+%T')] ${ep} → $code"
  done
  sleep "$INTERVAL"
done
