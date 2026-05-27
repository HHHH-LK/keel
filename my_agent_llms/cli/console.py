"""Shared rich.Console singleton.

All cli/ modules and chat.py import this same instance so that width / theme /
record state stay consistent across the app.
"""
from rich.console import Console

console = Console()
