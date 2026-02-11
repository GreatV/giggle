# Test Fixtures

Generate reference fixtures with librosa for cross-validation against Rust.

Usage:
```bash
python3 tests/data/generate_fixtures.py --out tests/data/fixtures --sr 22050
```

Notes:
- The script prefers the local librosa repo at `~/repos/librosa` if present.
- Outputs are `.npy` files plus `meta.txt` for reproducibility.
