"""Compact ball visibility intervals and noninteractive shading on Viser sliders."""

import html
import math

BALL_MAX_AGE = 0.1
LONG_GAP_SECONDS = 1.0


class BallVisibility:
  def __init__(self):
    self.intervals = []

  def add(self, start, end):
    """Append a visible time interval in chronological order; coalesce samples."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
      return
    if self.intervals and start <= self.intervals[-1][1] + 1e-9:
      self.intervals[-1][1] = max(end, self.intervals[-1][1])
    else:
      self.intervals.append([start, end])

  def gaps(self, start, end, minimum=LONG_GAP_SECONDS):
    """Complement visibility, including leading/trailing gaps, relative to start."""
    result, cursor = [], start
    for a, b in self.intervals:
      a, b = max(start, a), min(end, b)
      if b <= start or a >= end:
        continue
      if a - cursor >= minimum - 1e-9:
        result.append((cursor - start, a - start))
      cursor = max(cursor, b)
    if end - cursor >= minimum - 1e-9:
      result.append((cursor - start, end - start))
    return result


def gaps_from_samples(times, duration):
  visibility = BallVisibility()
  for t in sorted(times):
    visibility.add(float(t), float(t) + BALL_MAX_AGE)
  return visibility.gaps(0.0, duration)


class BallGapShading:
  """CSS over the existing slider track; preserves mouse/keyboard seek handlers."""

  def __init__(self, server, slider):
    self.slider = slider
    self.legend = server.gui.add_html("")

  def update(self, gaps, duration):
    stops, cursor = [], 0.0
    for start, end in gaps:
      a = min(100.0, max(0.0, start / max(duration, 1e-9) * 100))
      b = min(100.0, max(a, end / max(duration, 1e-9) * 100))
      stops.extend(
        [f"transparent {cursor:.6f}% {a:.6f}%", f"#343a40 {a:.6f}% {b:.6f}%"]
      )
      cursor = b
    stops.append(f"transparent {cursor:.6f}% 100%")
    # Mantine versions place the UUID either on the root or its input/thumb.
    uuid = self.slider._impl.uuid
    selector = f'.mantine-Slider-root:is([id="{uuid}"], :has([id="{uuid}"]))'
    total = sum(end - start for start, end in gaps)
    text = f"深色：连续 ≥{LONG_GAP_SECONDS:g} 秒无有效球观测 · {len(gaps)} 段，共 {total:.1f} 秒"
    self.legend.content = (
      f"<style>{selector} .mantine-Slider-track::after {{"
      "content: ''; position: absolute; inset: 0; z-index: 1; pointer-events: none;"
      f"background: linear-gradient(to right, {', '.join(stops)}); border-radius: inherit;"
      f"}} {selector} .mantine-Slider-thumb {{ z-index: 2; }}</style>"
      f'<span style="font-size:12px"><span style="color:#343a40">■</span> {html.escape(text)}</span>'
    )
