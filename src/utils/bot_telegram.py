"""
Notificador de Telegram para vigilar de forma remota el estado de los runs.

Uso:
    from bot_telegram import notify
    notify("Trial 3 terminado | best_loss=0.1234")

Rellena TELEGRAM_TOKEN a mano antes de ejecutar (abajo). El chat_id ya queda
puesto; cambialo si hace falta.

Diseno clave: notify() NUNCA lanza excepcion. Si la red se cae, el token esta
mal o Telegram no responde, lo registra y devuelve False, de modo que un fallo
de notificacion jamas tumbe un entrenamiento que ya esta corriendo.
"""
import logging

import requests

logger = logging.getLogger(__name__)

# --- Rellenar a mano antes de ejecutar -------------------------------------
TELEGRAM_TOKEN = "<token>"  # token del bot (te lo da BotFather)
TELEGRAM_CHAT_ID = "-5243358983"  # id del chat/grupo destino
# ---------------------------------------------------------------------------

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def notify(text: str, timeout: float = 10.0) -> bool:
    """
  Envia un mensaje de texto a Telegram.

  Args:
      text:    contenido del mensaje.
      timeout: segundos maximos de espera (evita que un envio colgado
               bloquee el run).

  Returns:
      True si Telegram acepto el mensaje, False si fallo o no hay token.
  """
    text = str(text).strip()
    if not text:
        return False

    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "<token>":
        # Sin token configurado: no es un error fatal, simplemente no notifica.
        logger.debug("TELEGRAM_TOKEN sin configurar; no se envia notificacion.")
        return False

    url = _API_URL.format(token=TELEGRAM_TOKEN)
    try:
        response = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=timeout,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("No se pudo enviar la notificacion de Telegram: %s", exc)
        return False


if __name__ == "__main__":
    # Prueba manual:  python bot_telegram.py
    logging.basicConfig(level=logging.INFO)
    print("Enviado" if notify("Prueba de notificacion desde bot_telegram.py")
          else "Fallo el envio (revisa token/chat_id/red)")

