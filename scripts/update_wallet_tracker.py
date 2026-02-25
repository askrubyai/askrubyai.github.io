#!/usr/bin/env python3
"""
Lightweight wallet tracker updater for homepage stats.

Usage:
  python3 scripts/update_wallet_tracker.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import requests

TRACKER_PATH = Path(__file__).resolve().parents[1] / "data" / "wallet-tracker.json"


def eth_call_balance(rpc_url: str, usdc_contract: str, wallet_address: str, decimals: int) -> float:
    wallet_hex = wallet_address.lower().removeprefix("0x")
    call_data = "0x70a08231" + wallet_hex.rjust(64, "0")  # balanceOf(address)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": usdc_contract, "data": call_data}, "latest"],
    }
    resp = requests.post(
        rpc_url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ruby-blog-wallet-tracker/1.0"
        },
        timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()

    if "error" in body:
        raise RuntimeError(f"RPC error: {body['error']}")

    raw = int(body["result"], 16)
    return raw / (10 ** decimals)


def main() -> None:
    data = json.loads(TRACKER_PATH.read_text())

    live_balance = eth_call_balance(
        rpc_url=data["rpcUrl"],
        usdc_contract=data["usdcContract"],
        wallet_address=data["walletAddress"],
        decimals=int(data.get("decimals", 6)),
    )

    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    history = data.setdefault("history", [])

    should_append = True
    if history:
        last = history[-1]
        last_bal = float(last.get("balance", 0.0))
        if abs(last_bal - live_balance) < 1e-9:
            # Same balance: just refresh timestamp/source to show latest sync.
            last["ts"] = now_iso
            last["source"] = "rpc-refresh"
            should_append = False

    if should_append:
        history.append({"ts": now_iso, "balance": round(live_balance, 6), "source": "rpc"})

    # Keep last 200 points (lightweight).
    data["history"] = history[-200:]

    TRACKER_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Updated tracker: balance=${live_balance:.6f} at {now_iso}")


if __name__ == "__main__":
    main()
