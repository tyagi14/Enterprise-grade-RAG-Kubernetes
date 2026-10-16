"""Test configuration: importing the gateway builds LLM clients, which need a (fake) key.
No test calls the network, so a placeholder lets the suite run without a .env (e.g. in CI)."""
import os

os.environ.setdefault("PORTKEY_API_KEY", "test-portkey-key")
os.environ.setdefault("PORTKEY_CONFIG_ID", "pc-test")
