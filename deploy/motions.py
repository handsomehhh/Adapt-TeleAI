"""Discover separately labelled original and physically corrected motion clips."""

from pathlib import Path


def motion_catalog(artifacts, selected):
  selected = Path(selected).resolve()
  sources = []
  if artifacts.run:
    sources += [
      ("训练原始动作", artifacts.run / "snapshot/dataset/g1_stage1/motions"),
      ("物理修正动作", artifacts.run / "corrected/motions"),
    ]
  if selected.parent not in [path.resolve() for _, path in sources]:
    sources.insert(0, ("指定目录", selected.parent))
  result = {}
  for source, directory in sources:
    for path in sorted(directory.glob("*.npz")):
      result[f"{source} / {path.name}"] = path.resolve()
  if selected not in result.values():
    raise FileNotFoundError(selected)
  return result
