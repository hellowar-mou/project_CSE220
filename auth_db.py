"""
Real authentication for the desktop app: SQLite-backed user store with
bcrypt password hashing. No plaintext passwords are ever stored or compared.
"""
import sqlite3
import bcrypt
import re
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _seed_demo_account():
    """Ensure a demo account exists so the app is usable immediately after clone."""
    conn = _connect()
    cur = conn.execute("SELECT 1 FROM users WHERE username = ?", ("signals",))
    if cur.fetchone() is None:
        pw_hash = bcrypt.hashpw("lti2026".encode(), bcrypt.gensalt())
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("signals", "signals@example.com", pw_hash),
        )
        conn.commit()
    conn.close()


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    pass


def validate_new_account(username, email, password, confirm_password):
    if len(username.strip()) < 3:
        raise AuthError("Username must be at least 3 characters.")
    if not EMAIL_RE.match(email.strip()):
        raise AuthError("Enter a valid email address.")
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters.")
    if password != confirm_password:
        raise AuthError("Passwords do not match.")


def register_user(username, email, password, confirm_password):
    validate_new_account(username, email, password, confirm_password)
    conn = _connect()
    try:
        cur = conn.execute("SELECT 1 FROM users WHERE username = ?", (username.strip(),))
        if cur.fetchone():
            raise AuthError("That username is already taken.")
        cur = conn.execute("SELECT 1 FROM users WHERE email = ?", (email.strip(),))
        if cur.fetchone():
            raise AuthError("An account with that email already exists.")

        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username.strip(), email.strip(), pw_hash),
        )
        conn.commit()
    finally:
        conn.close()


def verify_login(username, password):
    """Returns True/False. Uses bcrypt.checkpw — constant-time hash comparison."""
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username.strip(),)
        )
        row = cur.fetchone()
        if row is None:
            return False
        stored_hash = row[0]
        return bcrypt.checkpw(password.encode(), stored_hash)
    finally:
        conn.close()


# Seed the demo account on import so the app works immediately after a fresh clone.
_seed_demo_account()
