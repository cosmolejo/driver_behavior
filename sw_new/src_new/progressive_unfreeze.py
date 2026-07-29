"""
Descongelamiento progresivo (progressive / gradual unfreezing) del backbone.

Idea
----
Entrenar primero solo la cabeza (LSTM + atencion + clasificador) con el
backbone congelado, y liberar bloques del backbone de a poco, empezando por
los mas cercanos a la salida. Las capas tempranas de una CNN preentrenada
codifican caracteristicas genericas (bordes, texturas) que transfieren bien;
las tardias son especificas de ImageNet y son las que conviene readaptar.

Dos restricciones especificas de este proyecto
---------------------------------------------

1. BATCHNORM SIGUE CONGELADO SIEMPRE.

   Descongelar los PESOS de un bloque no es lo mismo que descongelar sus
   estadisticas de BatchNorm. El problema de batches temporalmente
   correlacionados (256 frames del mismo segmento por forward) es
   independiente de si los pesos se actualizan o no: mientras BN este en
   modo train, normaliza con estadisticas del segmento y filtra
   informacion.

   Este modulo NO toca el modo de BatchNorm. Requiere que el modelo se haya
   construido con `freeze_bn=True` y lo verifica explicitamente. Los
   gradientes siguen fluyendo a traves de BN usando las running_stats
   congeladas, que es el comportamiento correcto y la receta estandar para
   fine-tuning con batch efectivo chico.

2. LOS PARAM GROUPS SE CREAN UNA SOLA VEZ, AL PRINCIPIO.

   `OneCycleLR` captura los `max_lr` de cada param group en su constructor.
   Agregar grupos al optimizer despues de crearlo rompe el scheduler.

   Solucion: se crean TODOS los param groups desde el arranque (incluidos
   los que todavia estan congelados), cada uno con su learning rate
   diferenciado. El descongelamiento consiste unicamente en cambiar
   `requires_grad`. AdamW saltea los parametros cuyo `.grad` es None, asi
   que los grupos congelados no se actualizan ni acumulan weight decay
   mientras esten cerrados.

Uso tipico
----------
    unfreezer = ProgressiveUnfreezer(
        model,
        schedule={0: 0, 5: 2, 10: 4, 15: 8},   # epoca -> bloques finales
    )
    optimizer = torch.optim.AdamW(
        unfreezer.build_param_groups(base_lr=lr, decay=0.3),
        weight_decay=weight_decay,
    )
    scheduler = OneCycleLR(optimizer, max_lr=unfreezer.max_lrs(lr, decay=0.3), ...)

    for epoch in range(num_epochs):
        unfreezer.on_epoch_start(epoch)
        ...
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ProgressiveUnfreezer:
    """
    Parameters
    ----------
    model : nn.Module
        Debe exponer `feature_extractor` (los bloques del backbone) y haber
        sido construido con `freeze_bn=True`.
    schedule : dict {int: int}
        Mapea epoca -> cantidad de bloques FINALES del backbone que deben
        estar descongelados a partir de esa epoca. Debe ser monotona
        creciente. Ej: {0: 0, 5: 2, 10: 4} = solo cabeza hasta la epoca 4,
        ultimos 2 bloques desde la 5, ultimos 4 desde la 10.
    strict_bn : bool
        Si True (default), lanza error si el modelo no tiene freeze_bn=True.
    verbose : bool
        Imprime un resumen cada vez que se descongela un bloque nuevo.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: Dict[int, int],
        strict_bn: bool = True,
        verbose: bool = True,
    ):
        if not hasattr(model, "feature_extractor"):
            raise AttributeError("El modelo no expone `feature_extractor`.")

        freeze_bn = getattr(model, "freeze_bn", False)
        if strict_bn and not freeze_bn:
            raise ValueError(
                "El modelo debe construirse con freeze_bn=True. Descongelar "
                "pesos del backbone con BatchNorm en modo train reintroduce "
                "la fuga por normalizacion por segmento."
            )

        self.model = model
        self.verbose = verbose
        self.blocks: List[nn.Module] = list(model.feature_extractor.children())
        self.n_blocks = len(self.blocks)

        # --- Validacion del schedule ---
        if not schedule:
            raise ValueError("schedule vacio.")
        epochs = sorted(schedule)
        valores = [schedule[e] for e in epochs]
        if any(b < a for a, b in zip(valores, valores[1:])):
            raise ValueError(f"schedule debe ser monotono creciente: {valores}")
        if max(valores) > self.n_blocks:
            raise ValueError(
                f"schedule pide {max(valores)} bloques pero el backbone "
                f"tiene {self.n_blocks}."
            )
        self.schedule = {e: schedule[e] for e in epochs}

        # --- Etapas: cada salto del schedule es un param group propio ---
        # etapa 0 = cabeza (siempre entrenable)
        # etapa i = los bloques que se liberan en el i-esimo salto
        self._stage_bounds: List[tuple] = []
        anterior = 0
        for v in valores:
            if v > anterior:
                # bloques [n_blocks - v, n_blocks - anterior)
                self._stage_bounds.append((self.n_blocks - v, self.n_blocks - anterior))
                anterior = v

        self._unfrozen_now = -1  # fuerza el primer on_epoch_start a aplicar

        # Arranca todo el backbone congelado
        for p in self.model.feature_extractor.parameters():
            p.requires_grad = False

    # -----------------------------------------------------------------
    # Param groups
    # -----------------------------------------------------------------
    def _head_params(self) -> List[nn.Parameter]:
        """Todo lo que no es backbone: LSTM, atencion, clasificador, transformer."""
        backbone_ids = {id(p) for p in self.model.feature_extractor.parameters()}
        return [p for p in self.model.parameters() if id(p) not in backbone_ids]

    def _stage_params(self, stage_idx: int) -> List[nn.Parameter]:
        lo, hi = self._stage_bounds[stage_idx]
        params = []
        for b in range(lo, hi):
            params.extend(self.blocks[b].parameters())
        return params

    def build_param_groups(self, base_lr: float, decay: float = 0.3) -> List[dict]:
        """
        Un param group por etapa, con learning rate decreciente hacia las
        capas mas profundas (discriminative fine-tuning).

        grupo 0 (cabeza)  -> base_lr
        grupo 1           -> base_lr * decay
        grupo 2           -> base_lr * decay^2
        ...

        Las capas preentrenadas necesitan pasos mucho mas chicos que una
        cabeza inicializada al azar: con el mismo lr se destruirian las
        caracteristicas de ImageNet en las primeras iteraciones.
        """
        groups = [{"params": self._head_params(), "lr": base_lr, "name": "head"}]
        for i in range(len(self._stage_bounds)):
            lo, hi = self._stage_bounds[i]
            groups.append({
                "params": self._stage_params(i),
                "lr": base_lr * (decay ** (i + 1)),
                "name": f"backbone[{lo}:{hi}]",
            })
        return groups

    def max_lrs(self, base_lr: float, decay: float = 0.3) -> List[float]:
        """Lista de max_lr para OneCycleLR, alineada con build_param_groups()."""
        return [base_lr] + [
            base_lr * (decay ** (i + 1)) for i in range(len(self._stage_bounds))
        ]

    # -----------------------------------------------------------------
    # Descongelamiento
    # -----------------------------------------------------------------
    def n_unfrozen_at(self, epoch: int) -> int:
        """Cuantos bloques finales deben estar descongelados en esta epoca."""
        aplicable = [e for e in self.schedule if e <= epoch]
        return self.schedule[max(aplicable)] if aplicable else 0

    def on_epoch_start(self, epoch: int) -> bool:
        """
        Ajusta `requires_grad` segun el schedule. Devuelve True si hubo un
        cambio respecto de la epoca anterior.

        Llamar SIEMPRE despues de `model.train()`, no antes.
        """
        objetivo = self.n_unfrozen_at(epoch)
        if objetivo == self._unfrozen_now:
            return False

        primero = self.n_blocks - objetivo
        for i, blk in enumerate(self.blocks):
            for p in blk.parameters():
                p.requires_grad = i >= primero

        self._unfrozen_now = objetivo

        if self.verbose:
            entrenables = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.model.parameters())
            bb_entrenables = sum(
                p.numel() for p in self.model.feature_extractor.parameters()
                if p.requires_grad
            )
            bb_total = sum(p.numel() for p in self.model.feature_extractor.parameters())
            print(
                f"[unfreeze] epoca {epoch}: {objetivo}/{self.n_blocks} bloques "
                f"descongelados (indices {primero}..{self.n_blocks - 1})  |  "
                f"backbone entrenable: {bb_entrenables:,}/{bb_total:,} "
                f"({bb_entrenables / bb_total:.1%})  |  "
                f"modelo entrenable: {entrenables:,}/{total:,}"
            )
        return True

    # -----------------------------------------------------------------
    # Verificacion
    # -----------------------------------------------------------------
    def assert_bn_frozen(self):
        """
        Chequeo de seguridad: ninguna capa BatchNorm del backbone debe estar
        en modo train. Conviene llamarlo una vez por epoca durante las
        primeras corridas.
        """
        en_train = [
            name for name, m in self.model.feature_extractor.named_modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training
        ]
        if en_train:
            raise RuntimeError(
                f"{len(en_train)} capas BatchNorm en modo train "
                f"(ej: {en_train[:3]}). El fix de freeze_bn no se aplico."
            )

    def summary(self) -> str:
        lineas = [f"Backbone: {self.n_blocks} bloques", "Schedule:"]
        for e in sorted(self.schedule):
            n = self.schedule[e]
            if n == 0:
                lineas.append(f"  epoca {e:>3}: solo cabeza")
            else:
                lo = self.n_blocks - n
                params = sum(
                    p.numel() for b in range(lo, self.n_blocks)
                    for p in self.blocks[b].parameters()
                )
                bb_total = sum(p.numel() for p in self.model.feature_extractor.parameters())
                lineas.append(
                    f"  epoca {e:>3}: bloques {lo}..{self.n_blocks - 1} "
                    f"({params:,} params, {params / bb_total:.1%} del backbone)"
                )
        lineas.append("Param groups:")
        for i, (lo, hi) in enumerate(self._stage_bounds):
            lineas.append(f"  grupo {i + 1}: bloques [{lo}:{hi}]")
        return "\n".join(lineas)