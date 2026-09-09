#!/usr/bin/env python3
"""
can_context.py — Capa de contexto CAN para el sistema de alertado.

Extrae la logica de reglas de raspcan_sniffer.py y la envuelve en un lector
en hilo de fondo, de modo que el bucle de inferencia pueda consultar el
estado sin bloquearse.

Reglas
  G1a  V < V_off tras haber estado en marcha (histeresis)  -> suprime ambos
  G1b  VehicleSpeedVSOSigFailSts = 1                       -> suprime ambos
  G1c  0x146 sin recepcion durante T_stale                 -> suprime ambos
  G2   HazardLightSts = 1                                  -> suprime ambos
  R1   ReverseGearBCSts = 1, mas T_gracia                  -> suprime reaching
  R2   TurnIndicator en {Left, Right}, mas T_post          -> suprime reaching

La regla D (persistencia sobre N iteraciones) no vive aqui: opera sobre la
salida de los modelos y la implementa el pipeline.

Las condiciones se acumulan con OR entre dos llamadas consecutivas a
evaluate(): si una condicion fue cierta en cualquier instante del intervalo,
la iteracion asociada queda suprimida. Muestrear en un unico instante
descartaria transiciones intermedias.
"""

from __future__ import annotations

import threading
import time

# --------------------------------------------------------------------------
# Parametros de diseno
# --------------------------------------------------------------------------
# Ninguno procede del DBC. Son decisiones de diseno sin calibrar.

V_ON = 5.0   # km/h para declarar el vehiculo en marcha
V_OFF = 3.0  # km/h para declararlo detenido

T_REVERSE_GRACE = 1.5  # s de gracia tras salir de marcha atras
T_TURN_POST = 2.0      # s de permanencia tras apagar el intermitente
T_STALE = 0.5          # s sin recepcion para declarar una trama obsoleta

# Politica ante trama de contexto ausente (0x520 / 0x560). Con False, no poder
# leer el intermitente o la marcha atras no suprime reaching: se prefiere una
# falsa alarma a una ceguera silenciosa.
FAIL_SAFE_CONTEXT = False

# --- G1d: parada de trafico frente a vehiculo detenido ---------------------
# El start-stop solo apaga el motor con el vehiculo detenido EN CIRCULACION
# (semaforo, retencion), nunca aparcado. Permite distinguir dentro de G1 una
# parada momentanea de un vehiculo fuera de servicio.
#
# Con True se mantiene el comportamiento anterior: usar el telefono en un
# semaforo no genera alerta. Es una decision NORMATIVA, no tecnica: en varias
# jurisdicciones el uso del telefono sigue prohibido con el vehiculo detenido
# en via publica y el motor en marcha. Cambiar esta constante es la unica
# forma de alterar esa politica, y queda declarada en un solo sitio.
SUPPRESS_PHONE_AT_TRAFFIC_STOP = True

# --- M3 / M4: modulacion de sensibilidad por condiciones externas ----------
# Factores multiplicativos sobre el umbral de la media movil. Un factor < 1
# baja el umbral (mas sensible); 1.0 no tiene efecto.
#
# Ambos estan en 1.0 A PROPOSITO. El mecanismo queda cableado y las senales
# se registran, pero no hay forma de calibrar estos valores con los datos
# disponibles: DMD esta grabado de dia y sin lluvia, de modo que no existe
# ni un segmento con el que medir la degradacion. Fijarlos a ojo seria
# postular un efecto en lugar de medirlo. Se dejan neutros hasta disponer de
# captura en esas condiciones.
M3_RAIN_FACTOR = 1.0   # limpiaparabrisas activo
M4_NIGHT_FACTOR = 1.0  # luces de cruce encendidas

# --------------------------------------------------------------------------
# Tramas
# --------------------------------------------------------------------------

FRAME_SPEED = 0x146     # BRAKE_FE3       ciclo  20 ms
FRAME_VEHSTAT = 0x520   # VEHSTAT         ciclo  10 ms
FRAME_STATUS_B = 0x560  # STATUS_B_CAN    ciclo 100 ms

FRAME_IDS = (FRAME_SPEED, FRAME_VEHSTAT, FRAME_STATUS_B)

TURN_NAMES = {0: "off", 1: "left", 2: "right", 3: "sna"}
WIPER_SPEED_NAMES = {0: "off", 1: "low", 2: "high", 6: "rls_error", 7: "sna"}


def build_filters() -> list[dict]:
    """Un filtro por ID, mascara de 11 bits, tramas estandar.

    Los tres IDs son estandar. Un filtro con 'extended': True los descartaria
    todos, y python-can espera un entero en 'can_id', no una lista.
    """
    return [{"can_id": fid, "can_mask": 0x7FF, "extended": False} for fid in FRAME_IDS]


class ContextGate:
    """Evalua las reglas de supresion sobre un intervalo de tiempo."""

    def __init__(self) -> None:
        self.moving = False  # estado de la histeresis de velocidad
        self.last_speed = 0.0
        self.last_reverse = 0
        self.last_turn = 0
        self.last_hazard = 0
        self.last_fail = 0

        # G1d (0x146) y M3 / M4 (0x520)
        self.last_autostop_stay = 0    # 1 = RemainStop: motor parado por start-stop
        self.last_autostop_enable = 0  # 1 = Enable
        self.last_wiper_mode = 0       # 1 = Moving
        self.last_wiper_speed = 0      # 0 Off, 1 Low, 2 High, 6 RLS_Error, 7 SNA
        self.last_low_beam = 0         # 1 = ON

        # 'active' permite detectar el flanco de bajada. Sin el, la ventana de
        # gracia arrancaria en la primera trama recibida y suprimiria reaching
        # durante los primeros segundos sin motivo.
        self.reverse_active = False
        self.turn_active = False
        self.t_reverse_off: float | None = None
        self.t_turn_off: float | None = None

        self.last_rx = {fid: 0.0 for fid in FRAME_IDS}
        self._reset_window()

    def _reset_window(self) -> None:
        self.win_suppress_all = False
        self.win_suppress_reaching = False
        self.win_traffic_stop = False
        self.win_reasons: set[str] = set()

    # -- ingesta ------------------------------------------------------------

    def update(self, frame_id: int, decoded: dict, now: float) -> None:
        """Incorpora una trama y acumula las condiciones sobre el intervalo."""
        self.last_rx[frame_id] = now

        if frame_id == FRAME_SPEED:
            self.last_speed = decoded["VehicleSpeedVSOSig"]
            self.last_fail = decoded["VehicleSpeedVSOSigFailSts"]
            self.last_autostop_stay = decoded["NFRAutoStopStaySts"]
            self.last_autostop_enable = decoded["NFRAutoStopEnableSts"]

            # Histeresis: dos umbrales, no uno.
            if self.moving and self.last_speed < V_OFF:
                self.moving = False
            elif not self.moving and self.last_speed > V_ON:
                self.moving = True

            if self.last_fail:
                self.win_suppress_all = True
                self.win_reasons.add("G1b:fail")
            if not self.moving:
                # G1d: el start-stop solo actua con el vehiculo detenido en
                # circulacion, de modo que RemainStop identifica una parada
                # momentanea y no un vehiculo aparcado.
                traffic_stop = bool(self.last_autostop_stay)
                self.win_traffic_stop = self.win_traffic_stop or traffic_stop

                if traffic_stop and not SUPPRESS_PHONE_AT_TRAFFIC_STOP:
                    # Parada de trafico con la politica en "alertar": no se
                    # suprime phone, pero reaching si (girarse con el coche
                    # parado en un semaforo no es la maniobra que R1/R2 cubren,
                    # aunque tampoco hay riesgo de marcha).
                    self.win_suppress_reaching = True
                    self.win_reasons.add("G1d:traffic_stop")
                else:
                    self.win_suppress_all = True
                    self.win_reasons.add(
                        "G1d:traffic_stop" if traffic_stop else "G1a:stopped"
                    )

        elif frame_id == FRAME_VEHSTAT:
            self.last_turn = decoded["TurnIndicatorSts_CCAN1"]
            self.last_hazard = decoded["HazardLightSts"]
            self.last_wiper_mode = decoded["FrontWiperModeSts"]
            self.last_wiper_speed = decoded["RLSWiperSpeed"]
            self.last_low_beam = decoded["LowBeamSts_VEHSTAT"]

            if self.last_hazard:
                self.win_suppress_all = True
                self.win_reasons.add("G2:hazard")

            # SNA (3) se trata como Center: senal no disponible no suprime.
            if self.last_turn in (1, 2):
                self.win_suppress_reaching = True
                self.win_reasons.add("R2:turn_signal")
                self.turn_active = True
                self.t_turn_off = None
            elif self.turn_active:
                self.turn_active = False
                self.t_turn_off = now

        elif frame_id == FRAME_STATUS_B:
            self.last_reverse = decoded["ReverseGearBCSts"]

            if self.last_reverse:
                self.win_suppress_reaching = True
                self.win_reasons.add("R1:reverse")
                self.reverse_active = True
                self.t_reverse_off = None
            elif self.reverse_active:
                self.reverse_active = False
                self.t_reverse_off = now

    # -- decision -----------------------------------------------------------

    def evaluate(self, now: float) -> dict:
        """Cierra el intervalo y devuelve la decision. Reinicia el acumulador."""
        suppress_all = self.win_suppress_all
        suppress_reaching = self.win_suppress_reaching
        traffic_stop = self.win_traffic_stop
        reasons = set(self.win_reasons)

        # Obsolescencia. Se evalua al cerrar el intervalo, no en la ingesta,
        # porque la ausencia de trama no genera evento alguno.
        if now - self.last_rx[FRAME_SPEED] > T_STALE:
            suppress_all = True
            reasons.add("G1c:stale")
        for fid in (FRAME_VEHSTAT, FRAME_STATUS_B):
            if now - self.last_rx[fid] > T_STALE:
                reasons.add(f"ctx:{fid:#05x}_stale")
                if FAIL_SAFE_CONTEXT:
                    suppress_reaching = True

        # Ventanas de permanencia tras desactivarse la condicion.
        if self.t_reverse_off is not None and now - self.t_reverse_off < T_REVERSE_GRACE:
            suppress_reaching = True
            reasons.add("R1:grace")
        if self.t_turn_off is not None and now - self.t_turn_off < T_TURN_POST:
            suppress_reaching = True
            reasons.add("R2:post")

        # M3 / M4: modulacion por condiciones externas. Se calcula siempre y
        # se devuelve siempre, aunque los factores esten neutros: el objetivo
        # inmediato es dejar la covariable registrada junto a la prediccion
        # para poder medir la degradacion cuando existan datos de lluvia y de
        # noche, no alterar la decision a ciegas.
        raining = bool(self.last_wiper_mode) or self.last_wiper_speed in (1, 2)
        night = bool(self.last_low_beam)

        sensitivity = 1.0
        if raining:
            sensitivity *= M3_RAIN_FACTOR
            reasons.add("M3:rain")
        if night:
            sensitivity *= M4_NIGHT_FACTOR
            reasons.add("M4:low_beam")

        self._reset_window()
        return {
            "phone": not suppress_all,
            "reaching": not (suppress_all or suppress_reaching),
            "reasons": sorted(reasons),
            "speed": self.last_speed,
            "reverse": self.last_reverse,
            "turn": TURN_NAMES.get(self.last_turn, "?"),
            "hazard": self.last_hazard,
            # G1d
            "traffic_stop": int(traffic_stop),
            "autostop_enable": int(self.last_autostop_enable),
            # M3 / M4
            "raining": int(raining),
            "wiper_speed": WIPER_SPEED_NAMES.get(
                self.last_wiper_speed, str(self.last_wiper_speed)
            ),
            "low_beam": int(night),
            "sensitivity": sensitivity,
        }


class CanContextReader:
    """Lee el bus en un hilo de fondo y expone el estado ya evaluado.

    El hilo solo alimenta la puerta; la decision se toma cuando el bucle de
    inferencia llama a evaluate(), de modo que el intervalo de acumulacion
    coincide exactamente con una iteracion del pipeline.
    """

    def __init__(self, dbc_path: str, channel: str, interface: str = "socketcan"):
        import can
        import cantools

        self._can = can
        self.db = cantools.database.load_file(dbc_path, encoding="latin-1")
        self.bus = can.Bus(
            channel=channel, interface=interface, can_filters=build_filters()
        )

        self.gate = ContextGate()
        self.lock = threading.Lock()
        self.running = True
        self.rx_count = 0

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            try:
                msg = self.bus.recv(timeout=0.1)
            except Exception:  # el bus puede cerrarse durante el apagado
                break
            if msg is None or msg.arbitration_id not in FRAME_IDS:
                continue

            # decode_choices=False devuelve enteros. Con las etiquetas del DBC
            # activas cantools devuelve NamedSignalValue y una comparacion como
            # (valor == 1) es siempre falsa.
            decoded = self.db.decode_message(
                msg.arbitration_id, msg.data, decode_choices=False
            )
            with self.lock:
                self.gate.update(msg.arbitration_id, decoded, time.perf_counter())
                self.rx_count += 1

    def evaluate(self) -> dict:
        with self.lock:
            return self.gate.evaluate(time.perf_counter())

    def shutdown(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        try:
            self.bus.shutdown()
        except Exception:
            pass


class NullContextReader:
    """Sustituto sin CAN: ninguna regla suprime nada.

    Permite ejecutar el pipeline solo con vision y medir el efecto del filtro
    por diferencia, que es como conviene reportarlo en la memoria.
    """

    def evaluate(self) -> dict:
        return {
            "phone": True,
            "reaching": True,
            "reasons": [],
            "speed": float("nan"),
            "reverse": 0,
            "turn": "n/a",
            "hazard": 0,
            "traffic_stop": 0,
            "autostop_enable": 0,
            "raining": 0,
            "wiper_speed": "n/a",
            "low_beam": 0,
            "sensitivity": 1.0,
        }

    def shutdown(self) -> None:
        pass
