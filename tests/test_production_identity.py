from pathlib import Path
import hashlib

EXPECTED_SIZE = 78144
EXPECTED_SHA256 = "4628546a7c80ba5ce3212b7189f2708be14f7e5522cf875a5b915eed106f2b37"
EXPECTED_GIT_BLOB = "b797b26c9d34655b5fa49733e6c1c5d6f51265a4"

def test_production_source_identity():
    p = Path(__file__).resolve().parents[1] / "circle_arc_detector.py"
    data = p.read_bytes()
    assert len(data) == EXPECTED_SIZE
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256
    header = f"blob {len(data)}\0".encode()
    assert hashlib.sha1(header + data).hexdigest() == EXPECTED_GIT_BLOB
