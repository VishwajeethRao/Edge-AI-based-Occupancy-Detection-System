"""
EdgeVision-Lite / identity_memory.py

An anonymous identity store with two modes:

1. Session-only (default): a plain Python dict, cleared when the
   process exits. No file, no database, no network call.

2. Persistent (opt-in, via `db_path`): the SAME dict, but backed by a
   single local SQLite file so identities survive a restart. SQLite
   was chosen deliberately over a bigger database: it's part of the
   Python standard library (`sqlite3`, no new dependency), it's a
   single file on disk (no server process to run/manage), and it's
   more than sufficient for what this project needs to store --
   at most a few hundred embeddings, read/written a few times a
   second. No SQL is required from the user; all queries live inside
   this file.

What it stores per identity: a representative embedding vector, an
integer identity ID (P1, P2, ...), and first-seen / last-seen
timestamps. It never stores names, face crops, or any biometric
identity -- just an appearance embedding used for similarity matching.

MATCHING LOGIC
--------------
For a new embedding, compare it (cosine similarity) against every
stored identity's representative embedding (including ones loaded
from a previous session, if persistence is on). If the best match is
>= `threshold`, treat it as that returning person and refresh their
stored embedding (a light running average, so the representative
embedding adapts slowly to lighting/pose rather than freezing on the
very first glimpse). Otherwise, mint a brand-new anonymous identity.

This is approximate, similarity-based re-identification, not
biometric-grade recognition -- see the "Limitations" section of
README.md for when it can fail. Across a long gap between sessions
(different day, different clothing) it is *more* likely to fail than
within one session -- that tradeoff is inherent to appearance-only
Re-ID and is not solved by adding persistence.
"""

import sqlite3
import time

import numpy as np


class IdentityMemory:
    def __init__(self, threshold=0.6, ema_alpha=0.3, db_path=None):
        """
        threshold : cosine similarity required to call a new embedding
                    "the same person" as a stored identity. This is a
                    configurable STARTING VALUE (see README) -- it needs
                    validation against real video with your camera,
                    lighting, and clothing variety.
        ema_alpha : how much weight the newest embedding gets when
                    updating a matched identity's representative
                    embedding (exponential moving average). 0 = never
                    update, 1 = always overwrite with the latest look.
        db_path   : optional path to a SQLite file. If given, identities
                    persist across runs -- loaded on startup, written
                    through on every create/update. If None (default),
                    memory is session-only, exactly as before.
        """
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.identities = {}  # id -> {"embedding": np.ndarray, "first_seen": float, "last_seen": float, "hits": int}

        self.db_path = db_path
        self._conn = None
        if self.db_path:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    first_seen REAL,
                    last_seen REAL,
                    hits INTEGER
                )"""
            )
            self._conn.commit()
            self._load_from_db()

        self._next_id = (max(self.identities.keys()) + 1) if self.identities else 1

    def _load_from_db(self):
        cur = self._conn.execute("SELECT id, embedding, first_seen, last_seen, hits FROM identities")
        for row in cur.fetchall():
            ident_id, blob, first_seen, last_seen, hits = row
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            self.identities[ident_id] = {
                "embedding": emb,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "hits": hits,
            }

    def _persist(self, ident_id):
        if not self._conn:
            return
        rec = self.identities[ident_id]
        self._conn.execute(
            "INSERT OR REPLACE INTO identities (id, embedding, first_seen, last_seen, hits) VALUES (?, ?, ?, ?, ?)",
            (ident_id, rec["embedding"].astype(np.float32).tobytes(), rec["first_seen"], rec["last_seen"], rec["hits"]),
        )
        self._conn.commit()

    def _best_match(self, embedding):
        if not self.identities:
            return None, -1.0
        ids = list(self.identities.keys())
        mat = np.stack([self.identities[i]["embedding"] for i in ids])  # (K, 512), already L2-normalized
        sims = mat @ embedding  # cosine similarity since both sides are unit vectors
        best_idx = int(np.argmax(sims))
        return ids[best_idx], float(sims[best_idx])

    def match_or_create(self, embedding, frame_idx=0):
        """
        Returns (identity_id, is_new, similarity).
        Creates a new anonymous identity if nothing stored is similar
        enough; otherwise updates and returns the existing identity.
        `frame_idx` is kept for on-screen/debug use; persistence uses a
        wall-clock timestamp internally since frame numbers reset every run.
        """
        now = time.time()
        best_id, best_sim = self._best_match(embedding)

        if best_id is not None and best_sim >= self.threshold:
            rec = self.identities[best_id]
            rec["embedding"] = (1 - self.ema_alpha) * rec["embedding"] + self.ema_alpha * embedding
            norm = np.linalg.norm(rec["embedding"])
            if norm > 0:
                rec["embedding"] = rec["embedding"] / norm
            rec["last_seen"] = now
            rec["hits"] += 1
            self._persist(best_id)
            return best_id, False, best_sim

        new_id = self._next_id
        self._next_id += 1
        self.identities[new_id] = {
            "embedding": embedding.copy(),
            "first_seen": now,
            "last_seen": now,
            "hits": 1,
        }
        self._persist(new_id)
        return new_id, True, best_sim

    def close(self):
        if self._conn:
            self._conn.close()

    def __len__(self):
        return len(self.identities)
