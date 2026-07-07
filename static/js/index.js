document.addEventListener("DOMContentLoaded", () => {
  const copyButton = document.getElementById("copy-bibtex");
  const bibtexCode = document.getElementById("bibtex-code");

  if (!copyButton || !bibtexCode || !navigator.clipboard) {
    return;
  }

  copyButton.addEventListener("click", async () => {
    const originalText = copyButton.textContent;

    try {
      await navigator.clipboard.writeText(bibtexCode.textContent.trim());
      copyButton.textContent = "Copied";
    } catch (error) {
      copyButton.textContent = "Copy failed";
    }

    window.setTimeout(() => {
      copyButton.textContent = originalText;
    }, 1600);
  });
});
