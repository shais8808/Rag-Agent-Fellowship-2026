"""
auth.py
Real (if lightweight) user authentication — replaces the old name-only
simulated login gate.

Design goals, given this is a local/self-hosted Streamlit app:
  - No external auth provider required (works fully offline).
  - Passwords are never stored or compared in plaintext: PBKDF2-HMAC-SHA256
    with a random per-user salt (stdlib `hashlib`, no extra dependency).
  - Users persist across restarts in a local JSON file (mirrors how
    ChromaDB already persists to ./chroma_db — same "local file" model).
  - Deliberately simple: one flat users.json, no roles/permissions. Good
    enough for a small team / demo deployment; swap for a real identity
    provider (Auth0, Okta, Streamlit's built-in OIDC support, etc.) before
    putting this in front of the public internet.
"""

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from typing import Optional

USERS_FILE = "./users.json"
PBKDF2_ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


@dataclass
class AuthResult:
    ok: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """Returns (hash_hex, salt_hex). Generates a new random salt if none given."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def _verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    candidate_hash, _ = _hash_password(password, salt)
    # Constant-time comparison to avoid leaking hash info via timing.
    return secrets.compare_digest(candidate_hash, hash_hex)


# ---------------------------------------------------------------------------
# User store (local JSON file)
# ---------------------------------------------------------------------------

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    tmp_path = USERS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp_path, USERS_FILE)  # atomic on POSIX + Windows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_username(username: str) -> Optional[str]:
    if not USERNAME_RE.match(username):
        return "Username must be 3-32 characters: letters, numbers, underscore, dot, or hyphen."
    return None


def _validate_password(password: str) -> Optional[str]:
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if password.lower() == password or password.upper() == password:
        return "Password must mix upper and lower case letters."
    if not any(c.isdigit() for c in password):
        return "Password must include at least one number."
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sign_up(username: str, password: str, confirm_password: str) -> AuthResult:
    username = username.strip()

    err = _validate_username(username)
    if err:
        return AuthResult(False, err)

    if password != confirm_password:
        return AuthResult(False, "Passwords don't match.")

    err = _validate_password(password)
    if err:
        return AuthResult(False, err)

    users = _load_users()
    key = username.lower()
    if key in users:
        return AuthResult(False, "That username is already taken.")

    hash_hex, salt_hex = _hash_password(password)
    users[key] = {"display_name": username, "hash": hash_hex, "salt": salt_hex}
    _save_users(users)
    return AuthResult(True)


def log_in(username: str, password: str) -> AuthResult:
    username = username.strip()
    users = _load_users()
    record = users.get(username.lower())

    # Deliberately identical error for "no such user" and "wrong password" —
    # don't leak which usernames exist.
    if record is None or not _verify_password(password, record["hash"], record["salt"]):
        return AuthResult(False, "Incorrect username or password.")

    return AuthResult(True)


def get_display_name(username: str) -> str:
    users = _load_users()
    record = users.get(username.strip().lower())
    return record["display_name"] if record else username


def change_password(username: str, old_password: str, new_password: str, confirm_password: str) -> AuthResult:
    users = _load_users()
    key = username.strip().lower()
    record = users.get(key)
    if record is None or not _verify_password(old_password, record["hash"], record["salt"]):
        return AuthResult(False, "Current password is incorrect.")

    if new_password != confirm_password:
        return AuthResult(False, "New passwords don't match.")

    err = _validate_password(new_password)
    if err:
        return AuthResult(False, err)

    hash_hex, salt_hex = _hash_password(new_password)
    record["hash"], record["salt"] = hash_hex, salt_hex
    _save_users(users)
    return AuthResult(True)
