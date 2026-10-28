#!/usr/bin/env bash
cd /home2/ycj/Project/REVEAL
up() { env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY NO_PROXY='*' no_proxy='*' \
       curl -s -m 6 http://10.82.1.145:$1/v1/models 2>/dev/null | grep -q Qwen3.5-27B; }
while true; do
  if up 8001 && up 8004; then break; fi
  sleep 30
done
setsid bash scripts/variance_vmme.sh > outputs/logs/variance_vmme_driver.log 2>&1 &
echo "BOTH_ENDPOINTS_UP $(date '+%F %T') -> variance launched (setsid)"
