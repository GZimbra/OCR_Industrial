from pathlib import Path
from tempfile import gettempdir

from playwright.sync_api import sync_playwright


URL = "http://127.0.0.1:8000"
SCREENSHOT = Path(gettempdir()) / "ocr-industrial-ui.png"


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.goto(URL, wait_until="networkidle")
        page.get_by_role("heading", name="Quadros preparados para conferência.").wait_for()
        assert page.get_by_text("Operação local").is_visible()
        page.get_by_role("button", name="Revisão", exact=True).click()
        page.locator("#transcription").wait_for()
        assert page.locator(".preview-pane img").is_visible()
        page.get_by_role("button", name="Histórico", exact=True).click()
        page.locator(".data-table").wait_for()
        page.screenshot(path=str(SCREENSHOT), full_page=True)
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.goto(URL, wait_until="networkidle")
        assert mobile.get_by_role("button", name="Novo quadro", exact=True).is_visible()
        assert mobile.locator(".metrics").evaluate("e => getComputedStyle(e).gridTemplateColumns.split(' ').length") == 2
        browser.close()
    if errors:
        raise AssertionError(f"Erros no console: {errors}")
    print(f"ui-smoke: ok | screenshot={SCREENSHOT}")


if __name__ == "__main__":
    main()
