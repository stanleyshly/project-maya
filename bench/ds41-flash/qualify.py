#!/usr/bin/env python3
"""Exercise Maya's Anthropic API and print per-request qualification evidence."""

import argparse
import base64
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

MODEL = "deepseek-v4.1-flash"


class Client:
    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}"

    def post(self, path, body):
        request = urllib.request.Request(
            self.url + path, json.dumps(body).encode(),
            {"content-type": "application/json", "anthropic-version": "2023-06-01"},
        )
        return urllib.request.urlopen(request, timeout=14400)

    def call(self, label, messages, limit=64, thinking=False, **extra):
        body = dict(model=MODEL, messages=messages, max_tokens=limit, temperature=0,
                    stream=True, thinking=({"type": "enabled", "budget_tokens": 1024} if thinking
                                           else {"type": "disabled"}), **extra)
        start = time.monotonic()
        blocks = []
        first = finish = usage = None
        with self.post("/v1/messages", body) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                kind = event["type"]
                if kind == "content_block_start":
                    blocks.append(event["content_block"])
                elif kind == "content_block_delta":
                    if first is None:
                        first = time.monotonic()
                    delta = event["delta"]
                    for key in ("text", "thinking", "partial_json"):
                        if key in delta:
                            block = blocks[event["index"]]
                            block[key] = block.get(key, "") + delta[key]
                elif kind == "message_delta":
                    usage = event["usage"]
                    finish = event["delta"]["stop_reason"]
                elif kind == "error":
                    raise RuntimeError(event)
        if first is None or usage is None:
            raise RuntimeError(f"{label}: incomplete stream")
        for block in blocks:
            if "partial_json" in block:
                block["input"] = json.loads(block.pop("partial_json"))
        print(json.dumps(dict(label=label, ttft_s=first - start, elapsed_s=time.monotonic() - start,
                              finish=finish, usage=usage, content=blocks)), flush=True)
        return blocks, usage, finish

    def count(self, text):
        with self.post("/api/tokens", {"messages": [user(text)], "reasoning_effort": "none"}) as response:
            return json.load(response)["tokens"]


def user(text):
    return {"role": "user", "content": text}


def visible(blocks):
    return "".join(block["text"] for block in blocks if block["type"] == "text").strip()


def require(condition, description):
    if not condition:
        raise RuntimeError(description)


def image(color):
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (560, 560), color).save(buf, "PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                      "data": base64.b64encode(buf.getvalue()).decode()}}


def smoke(client):
    tools = [{"name": "read_color", "description": "Echoes the color of the supplied image.",
              "input_schema": {"type": "object", "properties": {"color": {"type": "string"}},
                               "required": ["color"]}}]
    history = [user([image("red"), {"type": "text", "text": "Use read_color with the image's color. "
                     "Then report the tool's exact result."}])]
    blocks, _, stop = client.call("image-tool", history, limit=2048, thinking=True, tools=tools)
    require(any(b["type"] == "thinking" and b["thinking"].strip() for b in blocks), "missing reasoning")
    calls = [b for b in blocks if b["type"] == "tool_use"]
    require(stop == "tool_use" and len(calls) == 1 and calls[0]["name"] == "read_color",
            "unexpected tool call")
    require(calls[0]["input"].get("color", "").lower() == "red", "incorrect image color")
    history += [{"role": "assistant", "content": blocks},
                user([{"type": "tool_result", "tool_use_id": calls[0]["id"], "content": "Crimson-8072"}])]
    blocks, _, _ = client.call("tool-result", history, limit=2048, thinking=True, tools=tools)
    require("Crimson-8072" in visible(blocks), "tool result did not reach final output")


def image_cache(client):
    def prompt(color):
        return user([image(color), {"type": "text", "text": " padded record." * 850
                                    + " What is the image's color? Reply with one word."}])

    red = prompt("red")
    first, _, _ = client.call("image-original", [red])
    require(visible(first).lower().rstrip(".") == "red", "incorrect original image color")
    client.call("image-interleave", [user(session(100) + "\nReply only OK.")])
    continuation = [{"role": "assistant", "content": first}, user("Repeat the color that you see.")]
    _, usage, _ = client.call("image-resume", [red, *continuation])
    require(usage["cache_read_input_tokens"] >= 1024, "matching image prefix did not resume")
    blocks, usage, _ = client.call("image-replaced", [prompt("blue"), *continuation])
    require(usage["cache_read_input_tokens"] < 64, "changed image reused stale state")
    require(visible(blocks).lower().rstrip(".") == "blue", "incorrect changed image color")


def session(index):
    return f"Case {index:02d} contains reference KITE-{index:02d}. " + " padded record." * 850


def slot_turn(client, index):
    blocks, usage, _ = client.call(f"resume-{index}", [
        user(session(index) + "\nReply only OK."), {"role": "assistant", "content": "OK"},
        user("Now reply with the case reference exactly and nothing else."),
    ])
    require(visible(blocks) == f"KITE-{index:02d}", f"case {index}: returned the wrong reference")
    require(usage["cache_read_input_tokens"] >= 1024 and usage["input_tokens"] < 128,
            f"case {index}: expected prefix reuse and a short suffix")


def slots(client):
    for index in range(10):
        blocks, _, _ = client.call(f"prime-{index}", [user(session(index) + "\nReply only OK.")])
        require(visible(blocks) == "OK", f"case {index}: unexpected priming response")
    for index in range(10):
        slot_turn(client, index)
    _, usage, _ = client.call("diverged", [user("New case, different reference. " + session(0)),
                                          user("Reply only OK.")])
    require(usage["cache_read_input_tokens"] < 16, "diverged case reused stale state")


def retrieval_prompt(client, target, lane):
    needle = f"ZEBRA-{target}-{lane}"
    lead = f"The secret retrieval code is {needle}.\n"
    tail = "\nReturn the secret retrieval code exactly, with no other text."
    filler = " oak maple pine elm birch ash cedar lime yew beech."
    base = client.count(lead + tail)
    step = 1024
    tokens_per_filler = (client.count(lead + filler * step + tail) - base) / step
    guess = int((target - base) / tokens_per_filler)
    lo, hi = max(0, guess - step), guess + step
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if client.count(lead + filler * mid + tail) <= target:
            lo = mid
        else:
            hi = mid
    text = lead + filler * lo + tail
    count = client.count(text)
    require(target - 32 <= count <= target, f"unexpected prompt length {count}")
    return text, needle


def retrieve(client, target, lane):
    text, needle = retrieval_prompt(client, target, lane)
    blocks, usage, _ = client.call(f"retrieve-{target}-{lane}", [user(text)], limit=256)
    require(target - 64 <= usage["input_tokens"] + usage["cache_read_input_tokens"] <= target + 32,
            "actual Anthropic prompt length differs from fixture")
    require(visible(blocks) == needle, f"lane {lane}: exact retrieval failed")


def archive(client, state_path):
    text, needle = retrieval_prompt(client, 523264, 0)
    blocks, usage, _ = client.call("archive-near-full", [user(text)], limit=256)
    require(visible(blocks) == needle, "near-full archive retrieval failed")
    state = {"text": text, "needle": needle, "prefix": usage["input_tokens"] + usage["cache_read_input_tokens"],
             "assistant": blocks}
    state_path.write_text(json.dumps(state))
    client.call("archive-interleave", [user(session(101) + "\nReply only OK.")])


def restore_long(client, state_path):
    state = json.loads(state_path.read_text())
    messages = [user(state["text"]), {"role": "assistant", "content": state["assistant"]},
                user("Return the same secret retrieval code exactly once again.")]
    blocks, usage, _ = client.call("restore-near-full", messages, limit=256)
    require(visible(blocks) == state["needle"], "restored conversation returned the wrong code")
    require(usage["cache_read_input_tokens"] >= state["prefix"] - 16 and usage["input_tokens"] < 128,
            "near-full restore did not reuse its prefix")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("smoke", "image-cache", "slots", "restore", "retrieval", "archive", "restore-long"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--case", type=int, default=9)
    parser.add_argument("--state", type=Path, help="scratch file for archive and restore-long")
    args = parser.parse_args()
    if args.check in ("archive", "restore-long") and args.state is None:
        parser.error("--state is required for this check")
    client = Client(args.port)
    if args.check == "smoke":
        smoke(client)
    elif args.check == "image-cache":
        image_cache(client)
    elif args.check == "slots":
        slots(client)
    elif args.check == "restore":
        slot_turn(client, args.case)
    elif args.check == "archive":
        archive(client, args.state)
    elif args.check == "restore-long":
        restore_long(client, args.state)
    else:
        for length in (262144, 523264):
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                jobs = [pool.submit(retrieve, client, length, lane) for lane in range(args.parallel)]
                for job in jobs:
                    job.result()


if __name__ == "__main__":
    main()
