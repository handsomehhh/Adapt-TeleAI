// Viser commands handle the bindings and editable-field filtering.
// Suppress key repeat/IME composition before those bindings receive the event.
(() => {
  if (window.__adaptG1KeyboardInstalled) return;
  window.__adaptG1KeyboardInstalled = true;
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (![' ', 'r', 'R'].includes(event.key)) return;
    if (event.target instanceof Element && event.target.closest(
      'input,textarea,select,[contenteditable="true"],[role="textbox"]'
    )) return;
    if (event.repeat || event.isComposing) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);
})();
