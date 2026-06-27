"""Redrob Intelligent Candidate Discovery & Ranking.

A staged retrieve-then-rank system:

  Stage 0 (offline)  precompute  — embeddings, features, consistency flags
  Stage 1 (online)   hard gates  — disqualifiers + honeypots
  Stage 2 (online)   retrieval   — semantic shortlist (~3k of 100k)
  Stage 3 (online)   core fit    — technical-gated T/S/E/C/L score
  Stage 4 (online)   behavioral  — 23 platform signals -> hireability
  Stage 5 (online)   trust       — consistency multiplier
  Stage 6 (online)   fusion      — final score, sort, top 100
  Stage 7 (online)   reasoning   — grounded, varied, rank-aware one-liners
"""

__version__ = "1.0.0"
