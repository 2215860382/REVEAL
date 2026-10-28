#!/usr/bin/env python3
"""Minimal LLM server using transformers, no vLLM dependency."""
import asyncio
import json
import torch
from aiohttp import web
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home2/hdl/models/Qwen3.5-27B"
PORT = 8004

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    print(f"Loading model from {MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    print("Model loaded and ready.", flush=True)

async def list_models(request):
    return web.json_response({
        "object": "list",
        "data": [{"id": "Qwen3.5-27B", "object": "model"}]
    })

async def chat_completions(request):
    data = await request.json()
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 2048)
    temperature = data.get('temperature', 0.0)

    # Convert messages to prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return web.json_response({
        "id": "chatcmpl-simple",
        "object": "chat.completion",
        "created": 0,
        "model": "Qwen3.5-27B",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": inputs['input_ids'].shape[1],
            "completion_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
            "total_tokens": outputs.shape[1]
        }
    })

if __name__ == '__main__':
    load_model()
    app = web.Application()
    app.router.add_get('/v1/models', list_models)
    app.router.add_post('/v1/chat/completions', chat_completions)
    web.run_app(app, host='0.0.0.0', port=PORT)
