"""
C.R.U.Y.F.F. — Ticket-Based WebSocket Auth (Fix F26)

Problem
-------
Fix F23 passed the JWT directly in the WebSocket URL query string:
``wss://...?token=<JWT>``.  This is an OWASP anti-pattern: URLs are
logged in plaintext by proxies, load balancers, and reverse proxy
access logs.  Anyone reading the stadium IT logs has the JWT.

Solution
--------
Ticket-based handshake:

1. Frontend POSTs to ``/ticket`` with the JWT in the Authorization
   header (never in the URL).
2. Backend validates the JWT, generates a cryptographically random
   opaque ticket, stores it with a 5-second TTL.
3. Frontend opens ``wss://...?ticket=<OPAQUE>``
4. Backend validates the ticket (one-time use, 5s TTL).
5. If the ticket is logged, it is already dead.

Dependencies
~~~~~~~~~~~~
* ``PyJWT`` — ``pip install pyjwt``
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import
_JWT_AVAILABLE = False
try:
    import jwt  # type: ignore[import-untyped]
    _JWT_AVAILABLE = True
except ImportError:
    pass

# Configuration
JWT_SECRET = os.environ.get("CRUYFF_JWT_SECRET", "cruyff-tactical-glass-secret-key")
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "cruyff-auth"
TICKET_TTL_SECONDS = 5
TICKET_LENGTH = 48  # bytes of entropy

# In-memory ticket store (in production, use Redis with TTL)
_ticket_store: dict[str, dict] = {}


# ─── JWT Operations ────────────────────────────────────────────

def create_match_token(match_id: str, ttl_hours: int = 4) -> str:
    """Generate a short-lived JWT for a specific match."""
    if not _JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed — pip install pyjwt")

    payload = {
        "sub": "tactical-glass",
        "match_id": match_id,
        "iss": JWT_ISSUER,
        "iat": int(time.time()),
        "exp": int(time.time()) + (ttl_hours * 3600),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _validate_jwt(token: str) -> dict:
    """Validate a JWT and return the decoded payload."""
    if not _JWT_AVAILABLE:
        logger.warning("PyJWT not installed — skipping validation (dev mode)")
        return {"sub": "tactical-glass", "dev_mode": True}

    try:
        return jwt.decode(
            token, JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")


# ─── Ticket Operations ─────────────────────────────────────────

def issue_ticket(jwt_token: str) -> str:
    """
    Exchange a valid JWT for a one-time, 5-second opaque ticket.

    Called by the ``POST /ticket`` endpoint.

    Parameters
    ----------
    jwt_token : str
        The JWT from the Authorization header.

    Returns
    -------
    str
        Opaque ticket string.

    Raises
    ------
    ValueError
        If the JWT is invalid.
    """
    # Validate the JWT first
    payload = _validate_jwt(jwt_token)

    # Generate cryptographically random ticket
    ticket = secrets.token_urlsafe(TICKET_LENGTH)

    # Store with TTL and payload
    _ticket_store[ticket] = {
        "payload": payload,
        "created": time.monotonic(),
        "used": False,
    }

    # Garbage-collect expired tickets
    _gc_tickets()

    logger.info(
        "Issued ticket for match=%s (expires in %ds)",
        payload.get("match_id", "?"),
        TICKET_TTL_SECONDS,
    )
    return ticket


def validate_ticket(ticket: Optional[str]) -> dict:
    """
    Validate a one-time ticket for WebSocket upgrade.

    Called during the WebSocket handshake.

    Parameters
    ----------
    ticket : str or None
        The opaque ticket from the query parameter.

    Returns
    -------
    dict
        Decoded JWT payload that was associated with the ticket.

    Raises
    ------
    ValueError
        If the ticket is missing, expired, already used, or unknown.
    """
    if not ticket:
        raise ValueError("Missing authentication ticket")

    entry = _ticket_store.get(ticket)
    if not entry:
        raise ValueError("Unknown or expired ticket")

    # One-time use
    if entry["used"]:
        raise ValueError("Ticket already consumed")

    # TTL check
    elapsed = time.monotonic() - entry["created"]
    if elapsed > TICKET_TTL_SECONDS:
        del _ticket_store[ticket]
        raise ValueError(f"Ticket expired ({elapsed:.1f}s > {TICKET_TTL_SECONDS}s)")

    # Mark as consumed and delete
    entry["used"] = True
    del _ticket_store[ticket]

    return entry["payload"]


def _gc_tickets() -> None:
    """Remove expired tickets from the store."""
    now = time.monotonic()
    expired = [
        k for k, v in _ticket_store.items()
        if (now - v["created"]) > TICKET_TTL_SECONDS * 2
    ]
    for k in expired:
        del _ticket_store[k]
