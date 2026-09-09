#!/usr/bin/env python3
"""
can_injector.py — Inyector de senales CAN artificiales sobre un bus virtual.

Emite ciclicamente las tres tramas que consume el filtro de contexto del DMS,
a sus tiempos de ciclo reales, y permite modificar en caliente las cinco
senales de interes desde una consola interactiva o desde un escenario en
fichero.

Uso tipico (requiere vcan0 activo):

    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan
    sudo ip link set up vcan0

    python3 can_injector.py                       # REPL sobre vcan0
    python3 can_injector.py --run aparcando       # escenario incorporado
    python3 can_injector.py --load mi_escenario.yaml

Para probar sin driver de kernel (dos procesos NO se ven entre si):

    python3 can_injector.py --interface virtual
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import can
import cantools

# --------------------------------------------------------------------------
# Definicion de tramas y senales
# --------------------------------------------------------------------------

# frame_id -> (nombre, periodo en segundos)  [GenMsgCycleTime del DBC]
FRAMES = {
    0x146: ("BRAKE_FE3", 0.020),
    0x520: ("VEHSTAT", 0.010),
    0x560: ("STATUS_B_CAN", 0.100),
}

# Valores de reposo coherentes con "vehiculo en marcha, sin incidencias".
# No se usan los valores iniciales del DBC porque varios de ellos son estados
# de fallo (VehicleSpeedVSOSigFailSts = Fail_present, KeySts = Data_not_valid),
# que dispararian las reglas de supresion nada mas arrancar.
BASELINE = {
    0x146: {"VehicleSpeedVSOSig": 0.0, "VehicleSpeedVSOSigFailSts": 0, "EPBSts": 0,
            "NFRAutoStopStaySts": 0, "NFRAutoStopEnableSts": 1},
    0x520: {"TurnIndicatorSts_CCAN1": 0, "HazardLightSts": 0, "KeySts": 4,
            "FrontWiperModeSts": 0, "RLSWiperSpeed": 0, "LowBeamSts_VEHSTAT": 0},
    0x560: {"ReverseGearBCSts": 0, "KeySts": 4, "DriverDoorSts": 0},
}

TURN_VALUES = {"off": 0, "center": 0, "left": 1, "right": 2, "sna": 3}
TURN_NAMES = {0: "off", 1: "left", 2: "right", 3: "sna"}
WIPER_VALUES = {"off": 0, "low": 1, "high": 2}

# comando del REPL -> (frame_id, nombre de senal en el DBC, conversor)
COMMANDS = {
    "speed": (0x146, "VehicleSpeedVSOSig", float),
    "fail": (0x146, "VehicleSpeedVSOSigFailSts", int),
    "epb": (0x146, "EPBSts", int),
    "autostop": (0x146, "NFRAutoStopStaySts", int),
    "turn": (0x520, "TurnIndicatorSts_CCAN1", lambda v: TURN_VALUES[str(v).lower()]),
    "hazard": (0x520, "HazardLightSts", int),
    "wiper": (0x520, "FrontWiperModeSts", int),
    "wiperspeed": (0x520, "RLSWiperSpeed", lambda v: WIPER_VALUES[str(v).lower()]),
    "lowbeam": (0x520, "LowBeamSts_VEHSTAT", int),
    "reverse": (0x560, "ReverseGearBCSts", int),
    "door": (0x560, "DriverDoorSts", int),
}

# --------------------------------------------------------------------------
# Escenarios incorporados
# --------------------------------------------------------------------------
# 't' es el instante absoluto en segundos desde el inicio del escenario.
# Ademas de senales, un paso admite 'drop' y 'resume' con un id de trama
# ('0x146') o 'all', para provocar y levantar una perdida de trama.

BUILTIN_SCENARIOS = {
    "circulando": {
        "description": "Steady drive at 60 km/h. No rule should suppress.",
        "steps": [
            {"t": 0.0, "speed": 0.0},
            {"t": 1.0, "speed": 30.0},
            {"t": 2.0, "speed": 60.0},
            {"t": 12.0, "speed": 60.0},
        ],
    },
    "aparcando": {
        "description": "Maneuver alternating D/R. Should suppress reaching.",
        "steps": [
            {"t": 0.0, "speed": 40.0},
            {"t": 3.0, "speed": 8.0},
            {"t": 5.0, "speed": 2.0, "reverse": 1},
            {"t": 9.0, "speed": 3.0, "reverse": 0},
            {"t": 11.0, "speed": 2.0, "reverse": 1},
            {"t": 15.0, "speed": 0.0, "reverse": 0},
        ],
    },
    "cambio_carril": {
        "description": "Right turn signal at cruising speed.",
        "steps": [
            {"t": 0.0, "speed": 90.0, "turn": "off"},
            {"t": 4.0, "turn": "right"},
            {"t": 8.0, "turn": "off"},
            {"t": 14.0, "speed": 90.0},
        ],
    },
    "semaforo": {
        "description": "Stop and go. Tests the speed hysteresis.",
        "steps": [
            {"t": 0.0, "speed": 50.0},
            {"t": 3.0, "speed": 10.0},
            {"t": 4.0, "speed": 4.0},
            {"t": 5.0, "speed": 0.0},
            {"t": 10.0, "speed": 2.0},
            {"t": 11.0, "speed": 4.0},
            {"t": 12.0, "speed": 20.0},
            {"t": 16.0, "speed": 45.0},
        ],
    },
    "perdida_senal": {
        "description": "Drops 0x146, then sets FailSts. Tests staleness handling.",
        "steps": [
            {"t": 0.0, "speed": 70.0},
            {"t": 4.0, "drop": "0x146"},
            {"t": 8.0, "resume": "0x146"},
            {"t": 12.0, "fail": 1},
            {"t": 16.0, "fail": 0},
            {"t": 20.0, "speed": 70.0},
        ],
    },
    "hazard": {
        "description": "Stopped on the shoulder with hazard lights on.",
        "steps": [
            {"t": 0.0, "speed": 80.0},
            {"t": 3.0, "speed": 20.0, "hazard": 1},
            {"t": 5.0, "speed": 0.0},
            {"t": 12.0, "hazard": 0},
        ],
    },
    "semaforo_startstop": {
        "description": "Traffic-light stop with start-stop engine off (G1d).",
        "steps": [
            {"t": 0.0, "speed": 45.0, "autostop": 0},
            {"t": 3.0, "speed": 8.0},
            {"t": 4.0, "speed": 0.0, "autostop": 1},
            {"t": 10.0, "autostop": 0},
            {"t": 11.0, "speed": 12.0},
            {"t": 13.0, "speed": 40.0},
        ],
    },
    "noche_lluvia": {
        "description": "Night drive in rain: exercises M3 and M4 covariates.",
        "steps": [
            {"t": 0.0, "speed": 70.0, "lowbeam": 1},
            {"t": 4.0, "wiper": 1, "wiperspeed": "low"},
            {"t": 8.0, "wiperspeed": "high"},
            {"t": 12.0, "wiper": 0, "wiperspeed": "off"},
            {"t": 16.0, "lowbeam": 0},
        ],
    },
}


# --------------------------------------------------------------------------
# Inyector
# --------------------------------------------------------------------------


class Injector:
    def __init__(self, dbc_path: str, channel: str, interface: str):
        self.db = cantools.database.load_file(dbc_path, encoding="latin-1")
        self.bus = can.Bus(channel=channel, interface=interface)

        self.messages = {fid: self.db.get_message_by_frame_id(fid) for fid in FRAMES}

        # Estado completo de cada trama: todas las senales a 0 fisico, luego
        # el baseline. Se necesita el diccionario completo porque encode()
        # exige un valor por senal.
        self.state = {}
        for fid, msg in self.messages.items():
            values = {s.name: 0 for s in msg.signals}
            values.update(BASELINE[fid])
            self.state[fid] = values

        self.enabled = {fid: True for fid in FRAMES}
        self.lock = threading.Lock()
        self.running = True
        self.tx_count = {fid: 0 for fid in FRAMES}

        self.scenario_thread: threading.Thread | None = None
        self.scenario_stop = threading.Event()

        self.threads = [
            threading.Thread(target=self._sender, args=(fid,), daemon=True)
            for fid in FRAMES
        ]
        for t in self.threads:
            t.start()

    # -- transmision ciclica ------------------------------------------------

    def _sender(self, fid: int) -> None:
        """Un hilo por trama. Periodo fijo, corregido por deriva."""
        msg = self.messages[fid]
        period = FRAMES[fid][1]
        next_tx = time.perf_counter()

        while self.running:
            next_tx += period
            sleep_for = next_tx - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # El hilo se ha quedado atras; resincroniza en vez de acumular.
                next_tx = time.perf_counter()

            with self.lock:
                if not self.enabled[fid]:
                    continue
                values = dict(self.state[fid])

            try:
                data = msg.encode(values)
                self.bus.send(
                    can.Message(arbitration_id=fid, data=data, is_extended_id=False)
                )
                with self.lock:
                    self.tx_count[fid] += 1
            except (can.CanError, cantools.database.EncodeError) as exc:
                print(f"\n[tx {hex(fid)}] error: {exc}", file=sys.stderr)

    # -- mutacion de estado -------------------------------------------------

    def set_signal(self, command: str, raw_value) -> str:
        if command not in COMMANDS:
            raise KeyError(command)
        fid, signal, convert = COMMANDS[command]
        value = convert(raw_value)

        sig = next(s for s in self.messages[fid].signals if s.name == signal)
        if sig.minimum is not None and value < sig.minimum:
            raise ValueError(f"{signal} < DBC minimum ({sig.minimum})")
        if sig.maximum is not None and value > sig.maximum:
            raise ValueError(f"{signal} > DBC maximum ({sig.maximum})")

        with self.lock:
            self.state[fid][signal] = value
        return f"{signal} = {value}  ({hex(fid)})"

    def set_frame_enabled(self, target: str, enabled: bool) -> str:
        targets = list(FRAMES) if target == "all" else [int(target, 0)]
        for fid in targets:
            if fid not in FRAMES:
                raise ValueError(f"unknown frame: {target}")
            with self.lock:
                self.enabled[fid] = enabled
        verb = "resumed" if enabled else "stopped"
        return f"transmission {verb}: {', '.join(hex(f) for f in targets)}"

    # -- escenarios ---------------------------------------------------------

    def run_scenario(self, scenario: dict, name: str = "") -> None:
        if self.scenario_thread and self.scenario_thread.is_alive():
            print("A scenario is already running. Use 'stop' first.")
            return

        self.scenario_stop.clear()
        self.scenario_thread = threading.Thread(
            target=self._run_scenario, args=(scenario, name), daemon=True
        )
        self.scenario_thread.start()

    def _run_scenario(self, scenario: dict, name: str) -> None:
        steps = sorted(scenario.get("steps", []), key=lambda s: float(s.get("t", 0.0)))
        start = time.perf_counter()
        print(f"\n>> scenario '{name}' started ({len(steps)} steps)")

        for step in steps:
            target = start + float(step.get("t", 0.0))
            while time.perf_counter() < target:
                if self.scenario_stop.wait(0.01):
                    print(">> scenario interrupted")
                    return

            applied = []
            for key, value in step.items():
                if key == "t":
                    continue
                try:
                    if key == "drop":
                        applied.append(self.set_frame_enabled(str(value), False))
                    elif key == "resume":
                        applied.append(self.set_frame_enabled(str(value), True))
                    else:
                        applied.append(self.set_signal(key, value))
                except (KeyError, ValueError) as exc:
                    applied.append(f"ERROR in '{key}': {exc}")

            elapsed = time.perf_counter() - start
            print(f"   t={elapsed:6.2f}s  " + " | ".join(applied))

        print(f">> scenario '{name}' finished\n")

    # -- introspeccion ------------------------------------------------------

    def describe_state(self) -> str:
        with self.lock:
            speed = self.state[0x146]["VehicleSpeedVSOSig"]
            fail = self.state[0x146]["VehicleSpeedVSOSigFailSts"]
            epb = self.state[0x146]["EPBSts"]
            turn = self.state[0x520]["TurnIndicatorSts_CCAN1"]
            hazard = self.state[0x520]["HazardLightSts"]
            reverse = self.state[0x560]["ReverseGearBCSts"]
            enabled = dict(self.enabled)
            counts = dict(self.tx_count)

        lines = [
            f"  velocidad   {speed:8.2f} km/h    (VehicleSpeedVSOSig, 0x146)",
            f"  fail speed  {fail:8d}         (VehicleSpeedVSOSigFailSts, 0x146)",
            f"  epb         {epb:8d}         (EPBSts, 0x146)",
            f"  intermit.   {TURN_NAMES.get(turn, turn):>8s}         (TurnIndicatorSts_CCAN1, 0x520)",
            f"  hazard      {hazard:8d}         (HazardLightSts, 0x520)",
            f"  reversa     {reverse:8d}         (ReverseGearBCSts, 0x560)",
            "",
        ]
        for fid, (fname, period) in FRAMES.items():
            flag = "activa " if enabled[fid] else "DETENIDA"
            lines.append(
                f"  {hex(fid)} {fname:<14s} {period * 1000:5.0f} ms  {flag}  "
                f"{counts[fid]} tramas enviadas"
            )
        return "\n".join(lines)

    def shutdown(self) -> None:
        self.running = False
        self.scenario_stop.set()
        for t in self.threads:
            t.join(timeout=0.5)
        self.bus.shutdown()


# --------------------------------------------------------------------------
# Consola
# --------------------------------------------------------------------------

HELP = """
Senales
  speed <km/h>            velocidad del vehiculo (0 - 511.9)
  reverse <0|1>           marcha atras
  turn <off|left|right|sna>   intermitente
  hazard <0|1>            luces de emergencia
  fail <0|1>              bandera de fallo de la senal de velocidad
  epb <0..7>              freno de mano (0=Released, 1=Applied)
  door <0|1>              puerta del conductor
  autostop <0|1>          start-stop: 1 = RemainStop (motor parado)  [G1d]
  wiper <0|1>             limpiaparabrisas: 1 = Moving               [M3]
  wiperspeed <off|low|high>   intensidad del sensor de lluvia        [M3]
  lowbeam <0|1>           luces de cruce                             [M4]

Tramas
  drop <0x146|0x520|0x560|all>     deja de transmitir (prueba obsolescencia)
  resume <0x146|...|all>           reanuda la transmision

Escenarios
  scenarios               lista los escenarios incorporados
  run <nombre>            ejecuta un escenario incorporado
  load <ruta>             ejecuta un escenario desde .yaml o .json
  stop                    interrumpe el escenario en curso

Otros
  state                   estado actual y contadores de transmision
  help                    esta ayuda
  quit                    salir
"""


def load_scenario_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        if path.lower().endswith(".json"):
            return json.load(fh)
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                "PyYAML no esta instalado. Usa un fichero .json o instala pyyaml."
            )
        return yaml.safe_load(fh)


def repl(inj: Injector) -> None:
    print("Inyector CAN activo. 'help' para la lista de comandos.\n")
    while True:
        try:
            line = input("can> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]

        try:
            if cmd in ("quit", "exit", "q"):
                return
            elif cmd in ("help", "h", "?"):
                print(HELP)
            elif cmd == "state":
                print(inj.describe_state())
            elif cmd == "scenarios":
                for name, sc in BUILTIN_SCENARIOS.items():
                    print(f"  {name:<16s} {sc['description']}")
            elif cmd == "run":
                name = args[0]
                if name not in BUILTIN_SCENARIOS:
                    print(f"Escenario desconocido: {name}")
                else:
                    inj.run_scenario(BUILTIN_SCENARIOS[name], name)
            elif cmd == "load":
                path = args[0]
                inj.run_scenario(load_scenario_file(path), os.path.basename(path))
            elif cmd == "stop":
                inj.scenario_stop.set()
            elif cmd in ("drop", "resume"):
                print(inj.set_frame_enabled(args[0], cmd == "resume"))
            elif cmd in COMMANDS:
                print(inj.set_signal(cmd, args[0]))
            else:
                print(f"Comando desconocido: {cmd}. Usa 'help'.")
        except IndexError:
            print(f"Faltan argumentos para '{cmd}'. Usa 'help'.")
        except (KeyError, ValueError) as exc:
            print(f"Error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbc", default="BODY_CAN.dbc")
    parser.add_argument("--channel", default="vcan0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--run", metavar="NOMBRE", help="escenario incorporado")
    parser.add_argument("--load", metavar="RUTA", help="escenario desde fichero")
    parser.add_argument(
        "--no-repl", action="store_true", help="no abrir consola tras el escenario"
    )
    args = parser.parse_args()

    inj = Injector(args.dbc, args.channel, args.interface)
    print(f"Transmitiendo en {args.channel} ({args.interface}).")

    try:
        if args.run:
            if args.run not in BUILTIN_SCENARIOS:
                raise SystemExit(f"Escenario desconocido: {args.run}")
            inj.run_scenario(BUILTIN_SCENARIOS[args.run], args.run)
        elif args.load:
            inj.run_scenario(load_scenario_file(args.load), os.path.basename(args.load))

        if args.no_repl:
            if inj.scenario_thread:
                inj.scenario_thread.join()
        else:
            repl(inj)
    finally:
        inj.shutdown()
        print("Inyector detenido.")


if __name__ == "__main__":
    main()
