"""Chinese labels for the same mjviser display controls used by PingPong."""

_LABELS = {
  "Frames": "坐标系",
  "Frame length": "坐标轴长度",
  "Frame width": "坐标轴宽度",
  "Enabled": "显示",
  "Color": "颜色",
  "Opacity": "不透明度",
  "Width": "宽度",
  "Height": "高度",
  "Inertia": "惯性",
  "Shape": "形状",
  "Joints": "关节",
  "Length": "长度",
  "Actuators": "执行器",
  "Auto-connect": "骨架连线",
  "Hide meshes": "隐藏网格",
  "Convex hull": "凸包",
  "Tendons": "肌腱",
  "COM": "质心",
  "Global scale": "整体缩放",
  "Geoms": "几何体",
  "Sites": "参考点",
}


def localize_model_controls(folder):
  # mjviser has no label hook. Only presentation is changed; dropdown values
  # remain its original enums so its callbacks continue to work unchanged.
  for child in tuple(folder._children.values()):
    label = getattr(child, "label", None)
    if label in ("Contact points", "Contact forces", "Constraints"):
      # These logs contain measured poses, not solved contact/constraint forces.
      child.remove()
      continue
    if label in _LABELS:
      child.label = _LABELS[label]
    if hasattr(child, "_children"):
      child.expand_by_default = False
      localize_model_controls(child)
