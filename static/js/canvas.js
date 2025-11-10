const tools = document.querySelectorAll('input[name="tool"]');

tools.forEach((tool) => {
  tool.addEventListener("change", () => {
    console.log(`Herramienta activa: ${tool.id}`);
    // Aquí podés cambiar el modo del canvas, por ejemplo
    if (tool.checked) {
      tools.forEach((t) => {
        if (t !== tool) t.checked = false;
      });
    }
  });
});
