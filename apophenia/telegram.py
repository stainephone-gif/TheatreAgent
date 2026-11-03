"""Уведомления художнику в Telegram и необязательное окно вето перед печатью."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

log = logging.getLogger("apophenia.telegram")


class Notifier:
    def __init__(self, cfg: Dict[str, Any]):
        t = cfg.get("telegram", {})
        self.token = os.getenv(t.get("token_env", "TELEGRAM_BOT_TOKEN"), "")
        self.chat_id = os.getenv(t.get("chat_id_env", "TELEGRAM_CHAT_ID"), "")
        self.enabled = bool(t.get("enabled", False)) and bool(self.token and self.chat_id)
        self.events = set(t.get("notify_on", []))
        self._offset: Optional[int] = None
        if bool(t.get("enabled", False)) and not self.enabled:
            log.warning("Telegram включён в конфиге, но токен или chat_id не заданы в .env")

    def _api(self, method: str, **params: Any) -> Dict[str, Any]:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}", json=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def send(self, text: str, event: Optional[str] = None) -> None:
        if not self.enabled or (event and event not in self.events):
            return
        try:
            self._api("sendMessage", chat_id=self.chat_id, text=text[:4000], disable_web_page_preview=True)
        except Exception as e:  # noqa: BLE001
            log.warning("Telegram: %s", e)

    def wait_for_veto(self, cycle_no: int, minutes: float) -> bool:
        """Окно модерации: True, если за `minutes` пришло сообщение «стоп» / «/stop» (с номером цикла или без)."""
        if not self.enabled:
            return False
        deadline = time.time() + minutes * 60
        while time.time() < deadline:
            try:
                data = self._api("getUpdates", timeout=20, offset=self._offset, allowed_updates=["message"])
                for upd in data.get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message", {})
                    if str(msg.get("chat", {}).get("id")) != str(self.chat_id):
                        continue
                    text = (msg.get("text") or "").strip().lower()
                    if text in ("стоп", "/stop", "stop") or text in (f"стоп {cycle_no}", f"/stop {cycle_no}", f"/stop_{cycle_no}"):
                        self.send(f"Цикл {cycle_no:05d}: печать отменена.")
                        return True
            except Exception as e:  # noqa: BLE001
                log.warning("Telegram getUpdates: %s", e)
                time.sleep(10)
        return False
