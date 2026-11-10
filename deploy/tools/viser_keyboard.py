"""Keep Viser's horizontal arrow keys in GUI controls, out of the camera."""


def protect_horizontal_arrow_keys(server):
  """Install for current and future clients without patching Viser assets.

  React handles sliders at its root before bubbling to body. Stop there,
  before Viser's document camera listener, without cancelling the control's
  default action. Keyup still reaches the camera to release any held key.
  RunJavascriptMessage is Viser's own buffered Plotly bootstrap mechanism.
  """
  from viser import _messages

  server._websock_server.queue_message(
    _messages.RunJavascriptMessage(
      source=r"""
(() => {
  if (window.__hopeHorizontalArrowGuard) return;
  window.__hopeHorizontalArrowGuard = true;
  document.body.addEventListener("keydown", (event) => {
    if (event.code === "ArrowLeft" || event.code === "ArrowRight" ||
        event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.stopPropagation();
    }
  });
})();
"""
    )
  )
