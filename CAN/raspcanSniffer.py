#!/usr/bin/env python3
"""
raspcan_sniffer.py — Filtro de contexto CAN para el sistema de deteccion de
distraccion del conductor.

Escucha las tres tramas necesarias, evalua las cinco reglas de supresion y
emite una decision por ventana de inferencia (0,5 s).

Reglas implementadas
  G1  velocidad bajo umbral, fallo de senal u obsolescencia -> suprime ambos
  G2  luces de emergencia activas                            -> suprime ambos
  R1  marcha atras (mas ventana de gracia)                   -> suprime reaching
  R2  intermitente activo (mas ventana posterior)            -> suprime reaching
  D   confirmacion temporal sostenida                        -> habilita alerta

Las condiciones se evaluan con OR sobre la ventana completa de inferencia: si
una condicion de supresion fue cierta en cualquier instante de los 0,5 s, la
decision asociada a esa ventana se suprime. Muestrear en un unico instante
descartaria transiciones intermedias (por ejemplo salir de marcha atras a
mitad de ventana durante una maniobra de aparcamiento).

Uso:
    python3 raspcan_sniffer.py                     # can0 real
    python3 raspcan_sniffer.py --channel vcan0     # contra el inyector

Puesta en marcha del bus real:
    sudo ip link set can0 down
    sudo ip link set can0 up type can bitrate 500000
    ip -details link show can0
"""

from __future__ import annotations

import argparse
import time

import can
import cantools

# --------------------------------------------------------------------------
# Parametros de diseno
# --------------------------------------------------------------------------
# Ninguno de estos valores procede del DBC. Son decisiones de diseno del
# sistema de alertado y deben documentarse como tales.

INFERENCE_WINDOW_S = 0.5  # 15 frames a 30 fps: latencia minima del modelo

# Histeresis de velocidad. V_OFF < V_ON evita el parpadeo de estado alrededor
# del punto de corte en trafico denso.
V_ON = 5.0  # km/h para declarar el vehiculo en marcha
V_OFF = 3.0  # km/h para declararlo detenido

# Ventanas temporales, expresadas como multiplos enteros de la ventana de
# inferencia: el sistema no puede reaccionar con mayor resolucion.
N_REVERSE_GRACE = 3  # 1,5 s tras salir de marcha atras
N_TURN_POST = 4  # 2,0 s tras apagar el intermitente
N_STALE = 1  # 0,5 s sin recepcion -> trama obsoleta
N_DWELL = 2  # 1,0 s de deteccion sostenida antes de alertar

T_REVERSE_GRACE = N_REVERSE_GRACE * INFERENCE_WINDOW_S
T_TURN_POST = N_TURN_POST * INFERENCE_WINDOW_S
T_STALE = N_STALE * INFERENCE_WINDOW_S

# Politica ante trama de contexto ausente (0x520 / 0x560). Con False, no poder
# leer el intermitente o la marcha atras no suprime reaching: se prefiere una
# falsa alarma a una ceguera silenciosa. Es una decision declarable.
FAIL_SAFE_CONTEXT = False

# --------------------------------------------------------------------------
# Tramas y senales
# --------------------------------------------------------------------------

FRAME_SPEED = 0x146  # BRAKE_FE3      ciclo  20 ms
FRAME_VEHSTAT = 0x520  # VEHSTAT      ciclo  10 ms
FRAME_STATUS_B = 0x560  # STATUS_B_CAN ciclo 100 ms

FRAME_IDS = (FRAME_SPEED, FRAME_VEHSTAT, FRAME_STATUS_B)

TURN_NAMES = {0: "off", 1: "left", 2: "right", 3: "sna"}


class ContextGate:
    """Evalua las reglas de supresion sobre una ventana de inferencia."""

    def __init__(self) -> None:
        self.moving = False  # estado de la histeresis de velocidad
        self.last_speed = 0.0
        self.last_reverse = 0
        self.last_turn = 0
        self.last_hazard = 0
        # 'active' permite detectar el flanco de bajada. Sin el, la ventana
        # de gracia arrancaria en la primera trama recibida y suprimiria
        # reaching durante los primeros segundos sin motivo.
        self.reverse_active = False
        self.turn_active = False
        self.t_reverse_off: float | None = None
        self.t_turn_off: float | None = None
        self.last_rx = {fid: 0.0 for fid in FRAME_IDS}

        self._reset_window()
        self.dwell = {"phone": 0, "reaching": 0}

    def _reset_window(self) -> None:
        self.win_suppress_all = False
        self.win_suppress_reaching = False
        self.win_reasons: set[str] = set()

    # -- ingesta ------------------------------------------------------------

    def update(self, frame_id: int, decoded: dict, now: float) -> None:
        """Incorpora una trama y acumula las condiciones sobre la ventana."""
        self.last_rx[frame_id] = now

        if frame_id == FRAME_SPEED:
            self.last_speed = decoded["VehicleSpeedVSOSig"]
            fail = decoded["VehicleSpeedVSOSigFailSts"]

            # Histeresis: dos umbrales, no uno.
            if self.moving and self.last_speed < V_OFF:
                self.moving = False
            elif not self.moving and self.last_speed > V_ON:
                self.moving = True

            if fail:
                self.win_suppress_all = True
                self.win_reasons.add("G1:fail")
            if not self.moving:
                self.win_suppress_all = True
                self.win_reasons.add("G1:detenido")

        elif frame_id == FRAME_VEHSTAT:
            self.last_turn = decoded["TurnIndicatorSts_CCAN1"]
            self.last_hazard = decoded["HazardLightSts"]

            if self.last_hazard:
                self.win_suppress_all = True
                self.win_reasons.add("G2:hazard")

            # SNA (3) se trata como Center: senal no disponible no suprime.
            if self.last_turn in (1, 2):
                self.win_suppress_reaching = True
                self.win_reasons.add("R2:intermitente")
                self.turn_active = True
                self.t_turn_off = None
            elif self.turn_active:
                self.turn_active = False
                self.t_turn_off = now

        elif frame_id == FRAME_STATUS_B:
            self.last_reverse = decoded["ReverseGearBCSts"]

            if self.last_reverse:
                self.win_suppress_reaching = True
                self.win_reasons.add("R1:reversa")
                self.reverse_active = True
                self.t_reverse_off = None
            elif self.reverse_active:
                self.reverse_active = False
                self.t_reverse_off = now

    # -- decision -----------------------------------------------------------

    def evaluate(self, now: float) -> dict:
        """Cierra la ventana y devuelve la decision. Reinicia el acumulador."""
        suppress_all = self.win_suppress_all
        suppress_reaching = self.win_suppress_reaching
        reasons = set(self.win_reasons)

        # Obsolescencia. Se evalua al cerrar la ventana, no en la ingesta,
        # porque la ausencia de trama no genera evento alguno.
        if now - self.last_rx[FRAME_SPEED] > T_STALE:
            suppress_all = True
            reasons.add("G1:obsoleta")
        for fid in (FRAME_VEHSTAT, FRAME_STATUS_B):
            if now - self.last_rx[fid] > T_STALE:
                reasons.add(f"ctx:{hex(fid)}_obsoleta")
                if FAIL_SAFE_CONTEXT:
                    suppress_reaching = True

        # Ventanas de permanencia tras desactivarse la condicion.
        if self.t_reverse_off is not None:
            if now - self.t_reverse_off < T_REVERSE_GRACE:
                suppress_reaching = True
                reasons.add("R1:gracia")
        if self.t_turn_off is not None:
            if now - self.t_turn_off < T_TURN_POST:
                suppress_reaching = True
                reasons.add("R2:posterior")

        self._reset_window()
        return {
            "phone": not suppress_all,
            "reaching": not (suppress_all or suppress_reaching),
            "reasons": sorted(reasons),
        }

    def confirm(self, model_output: str, gate: dict) -> str | None:
        """Regla D: exige N_DWELL ventanas consecutivas antes de alertar.

        'model_output' es la clase predicha por los modelos de vision en esta
        ventana: 'safe', 'phone' o 'reaching'. Devuelve la clase a alertar, o
        None. No consume ninguna senal CAN.
        """
        for cls in ("phone", "reaching"):
            if model_output == cls and gate[cls]:
                self.dwell[cls] += 1
            else:
                self.dwell[cls] = 0

        for cls in ("phone", "reaching"):
            if self.dwell[cls] >= N_DWELL:
                self.dwell[cls] = 0  # periodo refractario minimo
                return cls
        return None


def build_filters() -> list[dict]:
    """Un filtro por ID, mascara de 11 bits, tramas estandar.

    Los tres IDs son estandar. Un filtro con 'extended': True los descartaria
    todos, y python-can espera un entero en 'can_id', no una lista.
    """
    return [
        {"can_id": fid, "can_mask": 0x7FF, "extended": False} for fid in FRAME_IDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbc", default="BODY_CAN.dbc")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument(
        "--quiet", action="store_true", help="imprime solo los cambios de decision"
    )
    args = parser.parse_args()

    db = cantools.database.load_file(args.dbc, encoding="latin-1")
    bus = can.Bus(
        channel=args.channel, interface=args.interface, can_filters=build_filters()
    )
    gate = ContextGate()

    print(f"Escuchando {args.channel}: {', '.join(hex(f) for f in FRAME_IDS)}")
    print(f"Ventana de inferencia {INFERENCE_WINDOW_S * 1000:.0f} ms | "
          f"V_on {V_ON} / V_off {V_OFF} km/h\n")

    start = time.perf_counter()
    next_eval = start + INFERENCE_WINDOW_S
    last_line = ""

    try:
        while True:
            msg = bus.recv(timeout=0.05)
            now = time.perf_counter()

            if msg is not None and msg.arbitration_id in FRAME_IDS:
                # decode_choices=False devuelve enteros. Con las etiquetas del
                # DBC activas, cantools devuelve NamedSignalValue y una
                # comparacion como (valor == 1) es siempre falsa.
                decoded = db.decode_message(
                    msg.arbitration_id, msg.data, decode_choices=False
                )
                gate.update(msg.arbitration_id, decoded, now)

            if now < next_eval:
                continue
            next_eval += INFERENCE_WINDOW_S

            decision = gate.evaluate(now)
            line = (
                f"V={gate.last_speed:6.2f} km/h  "
                f"rev={gate.last_reverse}  "
                f"turn={TURN_NAMES.get(gate.last_turn, '?'):<5s}  "
                f"haz={gate.last_hazard}  ->  "
                f"phone={'ACTIVO   ' if decision['phone'] else 'SUPRIMIDO'}  "
                f"reaching={'ACTIVO   ' if decision['reaching'] else 'SUPRIMIDO'}"
            )
            reasons = " ".join(decision["reasons"])

            if args.quiet and line == last_line:
                continue
            last_line = line
            print(f"[{now - start:7.2f}s] {line}" + (f"   {reasons}" if reasons else ""))

    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()