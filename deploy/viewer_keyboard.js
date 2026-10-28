// Parent-page forwarding for the unified platform's cross-port iframe.
// Actual bindings and input-field filtering belong to Viser command hotkeys.
(() => {
  if (window.__pingpongKeyboardInstalled) return;
  window.__pingpongKeyboardInstalled = true;
  const editable = target => target instanceof Element &&
    !!target.closest('input,textarea,select,[contenteditable="true"],[role="textbox"]');
  document.addEventListener('keydown', event => {
    // Viser's camera tracks event.code, while command hotkeys use event.key.
    // Reserve replay arrows for seeking without feeding the camera's held keys.
    if (__REPLAY_MODE__ && !editable(event.target) && !event.ctrlKey &&
        !event.metaKey && !event.altKey && !event.shiftKey &&
        ['ArrowLeft', 'ArrowRight'].includes(event.key)) {
      Object.defineProperty(event, 'code', {value: ''});
    }
    if ((event.repeat || event.isComposing) && !editable(event.target)) event.stopImmediatePropagation();
  }, true);
  const origin = document.referrer ? new URL(document.referrer).origin : null;
  window.addEventListener('message', event => {
    if (window.parent === window || event.source !== window.parent ||
        event.origin !== origin || (event.data && event.data.type) !== 'pingpong-key') return;
    if (editable(document.activeElement)) return;
    document.body.dispatchEvent(new KeyboardEvent('keydown', {
      key: event.data.key, code: event.data.code, shiftKey: event.data.shiftKey,
      bubbles: true, cancelable: true
    }));
  });
})();
