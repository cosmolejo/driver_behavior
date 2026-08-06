"""
Data augmentation a nivel de CLIP para clasificación de comportamiento del
conductor.

Principio de diseño central
---------------------------
Todas las transformaciones fotométricas *sistemáticas* (brillo, contraste,
saturación, hue, gamma, blur) samplean sus parámetros UNA SOLA VEZ por clip
y se aplican idénticas a los T frames.

Motivo: si se sampleara un factor de brillo distinto por frame, se
introduciría un parpadeo que NO existe en el dominio real y que la LSTM
puede confundir con señal temporal legítima (un cambio de iluminación
frame-a-frame se parece a un cambio de escena). El objetivo de la
augmentation es simular variación *entre grabaciones*, no *dentro* de una.

Excepción deliberada: el ruido gaussiano SÍ se aplica de forma
independiente por frame (`per_frame_noise=True`), porque el ruido de sensor
real es temporalmente independiente. Aplicar el mismo patrón de ruido a
todos los frames sería el artefacto poco realista en este caso. Se deja
configurable por si se quiere lo contrario.

Layout de tensores
------------------
Las funciones aceptan clips en layout "CTHW" (C, T, H, W) — el mismo que
consume `MobileNetLSTMAttention.forward` — o "TCHW" (T, C, H, W). Se
convierte internamente a (T, C, H, W) porque los ops funcionales de
torchvision operan sobre (..., C, H, W) y así el batch temporal recibe
exactamente los mismos parámetros.

Rango esperado de entrada: float en [0, 1] (post-`ToTensor`, PRE-`Normalize`).
La normalización debe aplicarse DESPUÉS de la augmentation.
"""
from typing import Optional, Sequence, Tuple

import torch
from torchvision.transforms.v2 import functional as F


# ---------------------------------------------------------------------
# Helpers de layout
# ---------------------------------------------------------------------
def _to_tchw(clip: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "CTHW":
        return clip.permute(1, 0, 2, 3)
    if layout == "TCHW":
        return clip
    raise ValueError(f"layout debe ser 'CTHW' o 'TCHW', recibido: {layout!r}")


def _from_tchw(clip: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "CTHW":
        return clip.permute(1, 0, 2, 3)
    return clip


def _sample_uniform(rng: torch.Generator, low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(torch.empty(1).uniform_(low, high, generator=rng).item())


# ---------------------------------------------------------------------
# Augmentation fotométrica (prioridad 1)
# ---------------------------------------------------------------------
class ClipPhotometricAugment:
    """
    Augmentation fotométrica con parámetros fijos por clip.

    Cada parámetro `*_jitter` define un rango simétrico alrededor de 1.0
    (o de 0.0 para hue). Ej: `brightness=0.3` -> factor en [0.7, 1.3].

    Parameters
    ----------
    brightness, contrast, saturation : float
        Magnitud del jitter multiplicativo. 0 desactiva ese op.
    hue : float
        Magnitud del shift de hue, en [0, 0.5]. Se recomienda MUY chico
        (0.02-0.05): valores altos distorsionan tonos de piel de forma
        irreal y no aportan realismo en cabina.
    gamma : tuple(float, float) | None
        Rango de gamma. <1 aclara, >1 oscurece. None desactiva.
    noise_std : float
        Desvío estándar del ruido gaussiano (en escala [0,1]). 0 desactiva.
    blur_p : float
        Probabilidad de aplicar blur gaussiano al clip.
    blur_sigma : tuple(float, float)
        Rango de sigma del blur. Mantener bajo: un blur fuerte borra los
        detalles de mano/brazo que son la señal discriminativa.
    blur_kernel : int
        Tamaño de kernel del blur (impar).
    p : float
        Probabilidad de aplicar la augmentation completa al clip. Con p<1
        parte de los clips pasan sin modificar.
    per_frame_noise : bool
        Si True (default), el ruido se samplea independiente por frame
        (realista para ruido de sensor). Si False, el mismo patrón de ruido
        se repite en todos los frames.
    seed : int | None
        Semilla para reproducibilidad. None -> usa el RNG global de torch,
        que con `num_workers>0` en el DataLoader ya viene sembrado distinto
        por worker/época (comportamiento deseado).
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.03,
        gamma: Optional[Tuple[float, float]] = (0.8, 1.25),
        noise_std: float = 0.02,
        blur_p: float = 0.2,
        blur_sigma: Tuple[float, float] = (0.1, 1.0),
        blur_kernel: int = 5,
        p: float = 1.0,
        per_frame_noise: bool = True,
        layout: str = "CTHW",
        seed: Optional[int] = None,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.noise_std = noise_std
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.p = p
        self.per_frame_noise = per_frame_noise
        self.layout = layout

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        else:
            self._rng.seed()

    def __call__(self, clip: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Parameters
        ----------
        clip : Tensor
            (C, T, H, W) o (T, C, H, W) segun `self.layout`, float en [0,1].
        strength : float
            Multiplicador global de la intensidad de la augmentation.
            Permite augmentation class-aware (ver `ClassAwareAugment`).
            strength=0 -> identidad.
        """
        if strength <= 0 or _sample_uniform(self._rng, 0.0, 1.0) >= self.p:
            return clip

        original_dtype = clip.dtype
        x = _to_tchw(clip, self.layout).to(torch.float32)

        # --- Ops de color, en orden aleatorio (como hace ColorJitter) ---
        # Los factores se samplean UNA vez y valen para los T frames.
        ops = []

        if self.brightness > 0:
            b = self.brightness * strength
            factor = _sample_uniform(self._rng, max(0.0, 1 - b), 1 + b)
            ops.append(lambda t, f=factor: F.adjust_brightness(t, f))

        if self.contrast > 0:
            c = self.contrast * strength
            factor = _sample_uniform(self._rng, max(0.0, 1 - c), 1 + c)
            ops.append(lambda t, f=factor: F.adjust_contrast(t, f))

        if self.saturation > 0:
            s = self.saturation * strength
            factor = _sample_uniform(self._rng, max(0.0, 1 - s), 1 + s)
            ops.append(lambda t, f=factor: F.adjust_saturation(t, f))

        if self.hue > 0:
            h = min(0.5, self.hue * strength)
            factor = _sample_uniform(self._rng, -h, h)
            ops.append(lambda t, f=factor: F.adjust_hue(t, f))

        order = torch.randperm(len(ops), generator=self._rng).tolist()
        for i in order:
            x = ops[i](x)

        # --- Gamma (después del color, simula exposición de cámara) ---
        if self.gamma is not None:
            lo, hi = self.gamma
            # `strength` acerca/aleja el rango respecto de 1.0 (identidad)
            lo = 1.0 + (lo - 1.0) * strength
            hi = 1.0 + (hi - 1.0) * strength
            g = _sample_uniform(self._rng, lo, hi)
            x = F.adjust_gamma(x.clamp(0, 1), gamma=g)

        # --- Blur (mismo sigma para todo el clip) ---
        if self.blur_p > 0 and _sample_uniform(self._rng, 0.0, 1.0) < self.blur_p * strength:
            sigma = _sample_uniform(self._rng, *self.blur_sigma)
            x = F.gaussian_blur(x, kernel_size=[self.blur_kernel] * 2, sigma=[sigma, sigma])

        # --- Ruido (independiente por frame: ver docstring del módulo) ---
        if self.noise_std > 0:
            std = self.noise_std * strength
            if self.per_frame_noise:
                noise = torch.randn(x.shape, generator=self._rng, dtype=x.dtype)
            else:
                frame_noise = torch.randn(
                    (1,) + tuple(x.shape[1:]), generator=self._rng, dtype=x.dtype
                )
                noise = frame_noise.expand_as(x)
            x = x + noise * std

        x = x.clamp(0, 1).to(original_dtype)
        return _from_tchw(x, self.layout)


# ---------------------------------------------------------------------
# Augmentation temporal (siguiente iteración — off por defecto)
# ---------------------------------------------------------------------
class ClipTemporalCutout:
    """
    Enmascara algunos frames aislados del clip, simulando frames
    perdidos/corruptos en el stream.

    Más seguro que un cutout espacial: no arriesga ocluir permanentemente
    la región de mano/brazo (que es la señal discriminativa), sino que
    elimina información puntual en el eje temporal.

    mode:
        "repeat" -> reemplaza el frame por el anterior (más realista:
                    así se ve un frame droppeado en un decoder real)
        "zero"   -> pone el frame en negro
    """

    def __init__(
        self,
        num_frames: int = 2,
        p: float = 0.3,
        mode: str = "repeat",
        layout: str = "CTHW",
        seed: Optional[int] = None,
    ):
        if mode not in ("repeat", "zero"):
            raise ValueError(f"mode debe ser 'repeat' o 'zero', recibido: {mode!r}")
        self.num_frames = num_frames
        self.p = p
        self.mode = mode
        self.layout = layout

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        else:
            self._rng.seed()

    def __call__(self, clip: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if strength <= 0 or _sample_uniform(self._rng, 0.0, 1.0) >= self.p * strength:
            return clip

        x = _to_tchw(clip, self.layout).clone()
        T = x.shape[0]
        k = min(self.num_frames, max(0, T - 1))
        if k == 0:
            return clip

        # Nunca se enmascara el frame 0 (no hay "anterior" del cual copiar)
        idx = torch.randperm(T - 1, generator=self._rng)[:k] + 1
        for i in idx.tolist():
            if self.mode == "repeat":
                x[i] = x[i - 1]
            else:
                x[i] = 0.0

        return _from_tchw(x, self.layout)


def sample_window_indices(
    num_available_frames: int,
    sequence_length: int,
    train: bool,
    stride_jitter: Sequence[int] = (1, 2),
    rng: Optional[torch.Generator] = None,
) -> list:
    """
    Selección de índices de frames para una ventana, con jitter de
    frame-rate y offset temporal aleatorio.

    IMPORTANTE: esto NO se aplica sobre un clip ya formado — hay que
    llamarlo en el punto de `SegmentDataset` donde se decide QUÉ frames
    componen cada ventana. Es la pieza de augmentation temporal que
    requiere tocar la construcción de ventanas, por eso queda separada
    de las clases de arriba.

    - `stride_jitter`: strides posibles. stride=2 toma 1 de cada 2 frames,
      simulando una grabación a menor FPS (o un movimiento más rápido).
    - offset: desde qué frame arranca la ventana, si sobran frames.
    - En validación/test (`train=False`) devuelve siempre stride=1 y
      offset centrado, para que la evaluación sea determinista.

    Si no hay frames suficientes, se hace padding repitiendo el último
    índice disponible (misma política que suele usarse para segmentos
    cortos).
    """
    if rng is None:
        rng = torch.Generator()
        rng.seed()

    if not train:
        stride = 1
        span = min(sequence_length, num_available_frames)
        offset = max(0, (num_available_frames - span) // 2)
    else:
        valid_strides = [
            s for s in stride_jitter
            if (sequence_length - 1) * s + 1 <= num_available_frames
        ] or [1]
        stride = valid_strides[
            int(torch.randint(len(valid_strides), (1,), generator=rng).item())
        ]
        span = (sequence_length - 1) * stride + 1
        max_offset = max(0, num_available_frames - span)
        offset = (
            int(torch.randint(max_offset + 1, (1,), generator=rng).item())
            if max_offset > 0 else 0
        )

    idx = [offset + i * stride for i in range(sequence_length)]
    last_valid = num_available_frames - 1
    return [min(i, last_valid) for i in idx]


# ---------------------------------------------------------------------
# Augmentation class-aware
# ---------------------------------------------------------------------
class ClassAwareAugment:
    """
    Aplica una o más augmentations con intensidad dependiente de la clase.

    Sirve como oversampling implícito barato: la clase minoritaria
    (`reaching`) recibe transformaciones más agresivas, generando más
    variedad efectiva sin duplicar frames idénticos ni tocar la loss.

    Parameters
    ----------
    transforms : lista de callables
        Cada uno debe aceptar `(clip, strength)`.
    strength_by_class : dict {label_int: float}
        Multiplicador de intensidad por clase. Default 1.0 para las
        clases no listadas.

    Ejemplo (0=safe, 1=reaching, 2=unsafe):
        ClassAwareAugment(
            [ClipPhotometricAugment()],
            strength_by_class={0: 0.8, 1: 1.3, 2: 1.0},
        )
    """

    def __init__(self, transforms: Sequence, strength_by_class: Optional[dict] = None):
        self.transforms = list(transforms)
        self.strength_by_class = dict(strength_by_class or {})

    def __call__(self, clip: torch.Tensor, label: Optional[int] = None) -> torch.Tensor:
        strength = self.strength_by_class.get(int(label), 1.0) if label is not None else 1.0
        for t in self.transforms:
            clip = t(clip, strength=strength)
        return clip


# ---------------------------------------------------------------------
# Constructor de conveniencia
# ---------------------------------------------------------------------
def build_clip_augment(
    train: bool,
    num_classes: int = 3,
    strength_by_class: Optional[dict] = None,
    use_photometric: bool = True,
    use_temporal_cutout: bool = False,
    temporal_cutout_frames: int = 2,
    layout: str = "CTHW",
    seed: Optional[int] = None,
) -> Optional["ClassAwareAugment"]:
    """
    Arma el pipeline de augmentation para train, o None para val/test.

    El jitter de frame-rate y el offset temporal NO se arman aca: no operan
    sobre un clip ya cargado sino sobre QUE frames se cargan, asi que viven
    en SegmentDataset (parametros `temporal_stride_jitter` y
    `temporal_offset_jitter`). Aca solo van las transformaciones que se
    aplican al tensor del clip.
    """
    if not train:
        return None

    transforms_list = []

    if use_photometric:
        transforms_list.append(
            ClipPhotometricAugment(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.03,
                gamma=(0.8, 1.25),
                noise_std=0.02,
                blur_p=0.2,
                blur_sigma=(0.1, 1.0),
                p=0.8,
                layout=layout,
                seed=seed,
            )
        )

    if use_temporal_cutout and temporal_cutout_frames > 0:
        transforms_list.append(
            ClipTemporalCutout(
                num_frames=temporal_cutout_frames,
                p=0.3,
                mode="repeat",
                layout=layout,
                seed=seed,
            )
        )

    if not transforms_list:
        return None

    return ClassAwareAugment(transforms_list, strength_by_class=strength_by_class)