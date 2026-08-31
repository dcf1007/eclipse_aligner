"""Superseded compatibility entry point for the final synchronous threshold flow.

The deferred threshold-refinement architecture was intentionally replaced. Running
this historical entry point now applies the canonical agreed transformation instead.
"""
from apply_agreed_threshold_unification import main


if __name__ == "__main__":
    main()
