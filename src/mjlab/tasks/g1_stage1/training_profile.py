"""Optional, immutable G1 training settings; original tennis never imports this."""

import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def read_profile(path):
  value = json.loads(Path(path).read_text())
  if value.get("schema") != "g1-adapt-training-profile-v1":
    raise ValueError("Unknown G1 training profile schema")
  return value


def training_profile():
  path = os.environ.get("ADAPT_G1_TRAINING_PROFILE")
  return read_profile(str(Path(path).resolve())) if path else {}
