"""Single source of truth for all CLI colors.

Change a value here to repaint the whole UI. No other module in cli/ or
chat.py is allowed to hard-code a color string.
"""

# ── Roles ────────────────────────────────────────────────
YOU      = "cyan"              # user message left bar + role label
AGENT    = "magenta"           # agent reply left bar + role label

# ── Accents ──────────────────────────────────────────────
ACCENT   = "bright_magenta"    # ❯ arrow, table headers, completion-menu highlight bg
LOGO_L   = "magenta"           # banner left gradient block (█)
LOGO_R   = "bright_magenta"    # banner right gradient block (▒)
TITLE    = "bold cyan"         # banner title

# ── States ───────────────────────────────────────────────
OK       = "green"
WARN     = "yellow"
ERR      = "red"

# ── Neutral ──────────────────────────────────────────────
DIM      = "bright_black"      # meta, timestamps, descriptions
RULE     = "bright_black"      # horizontal separator
DEFAULT  = ""                  # keep terminal default fg
