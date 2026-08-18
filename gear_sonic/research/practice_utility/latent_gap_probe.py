"""LUCID's latent command-execution gap, carried forward as a *probe*.

LUCID v1 measured the mismatch between the joint targets a policy commands and
the joint motion actually realized, in the latent space of a temporal VAE
pre-trained on reference motion, and used a high quantile of that gap to drive
a PI controller over a global domain-randomization scale.

Here the same signal is kept but its job changes. It is a **proxy and mechanism
diagnostic**, never the scheduler. Two reasons:

* Scoring a curriculum with the same quantity that drives it makes any
  improvement partly definitional. The gap is one of the candidate predictors
  being *audited* against measured practice utility, so it must sit on the
  predictor side of the ledger, not the outcome side.
* A gap can improve while reward and episode length degrade. Treating it as an
  outcome would let that trade look like progress.

Why latent rather than raw joint space: contact transients -- foot impacts
above all -- produce large, brief excursions in raw joint error that say
nothing about whether the commanded behaviour was realizable. An encoder
trained to reconstruct real motion has no capacity allocated to such spikes, so
they are attenuated while sustained deviation survives. :func:`gap_series`
computes both, and ``test_latent_gap_probe`` verifies the attenuation on a
trained encoder rather than assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

#: Gap assigned to timesteps after an episode terminates, so terminated and
#: completed episodes remain comparable over a fixed horizon (LUCID v1 uses the
#: cosine-distance maximum, 2.0).
TERMINATED_GAP = 2.0


@dataclass(frozen=True)
class WindowSpec:
    """Shape of the command/execution windows fed to the encoder."""

    length: int = 16
    stride: int = 1

    def __post_init__(self) -> None:
        if self.length < 2:
            raise ValueError(f"window length must be >= 2, got {self.length}")
        if self.stride < 1:
            raise ValueError(f"window stride must be >= 1, got {self.stride}")

    @property
    def span(self) -> int:
        """Timesteps spanned, i.e. how long warmup lasts."""
        return (self.length - 1) * self.stride + 1


class TemporalVAE(nn.Module):
    """Small temporal VAE over ``(window_length, num_joints)`` motion windows.

    Convolutional along time so the encoder sees temporal structure rather than
    a bag of frames. Deliberately small: it is a fixed measuring instrument, and
    a large one would risk memorizing the very perturbations it should attenuate.
    """

    def __init__(
        self,
        num_joints: int,
        window_length: int = 16,
        latent_dim: int = 32,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if num_joints < 1:
            raise ValueError(f"num_joints must be >= 1, got {num_joints}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        self.num_joints = num_joints
        self.window_length = window_length
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(num_joints, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        flat = hidden_channels * window_length
        self.to_mu = nn.Linear(flat, latent_dim)
        self.to_logvar = nn.Linear(flat, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat),
            nn.GELU(),
            nn.Unflatten(1, (hidden_channels, window_length)),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, num_joints, kernel_size=5, padding=2),
        )

    def encode(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` for a batch of ``(B, T, J)`` windows."""
        features = self.encoder(self._check(windows).transpose(1, 2))
        logvar = self.to_logvar(features).clamp(-10.0, 10.0)
        return self.to_mu(features), logvar

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent).transpose(1, 2)

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(windows)
        latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(latent), mu, logvar

    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        """Deterministic embedding: the posterior mean, no sampling.

        The gap must be a function of the motion alone. Sampling here would make
        the same pair of windows yield different gaps on repeated evaluation.
        """
        with torch.no_grad():
            return self.encode(windows)[0]

    def _check(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError(f"windows must be (B, T, J), got {tuple(windows.shape)}")
        if windows.shape[1] != self.window_length:
            raise ValueError(
                f"window length {windows.shape[1]} != encoder's {self.window_length}"
            )
        if windows.shape[2] != self.num_joints:
            raise ValueError(f"joint count {windows.shape[2]} != encoder's {self.num_joints}")
        return windows


def elbo_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gaussian-decoder ELBO: squared reconstruction error plus beta*KL."""
    recon = (reconstruction - target).pow(2).sum(dim=(1, 2)).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return recon + beta * kl, {"recon": float(recon), "kl": float(kl)}


def corrupt_windows(
    windows: torch.Tensor,
    noise_std: float = 0.01,
    spike_prob: float = 0.05,
    spike_scale: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add sensor noise and brief transients to a batch of windows.

    Training the encoder to reconstruct the *clean* window from a corrupted one
    is what teaches it to treat a one-frame spike as something to remove rather
    than represent. That denoising behaviour is the entire reason the latent gap
    is steadier than raw joint error under contact.
    """
    if noise_std < 0 or spike_scale < 0:
        raise ValueError("noise_std and spike_scale must be non-negative")
    if not 0.0 <= spike_prob <= 1.0:
        raise ValueError(f"spike_prob must be in [0, 1], got {spike_prob}")

    kwargs = {"generator": generator} if generator is not None else {}
    noisy = windows + torch.randn(windows.shape, **kwargs).to(windows) * noise_std
    if spike_prob > 0:
        mask = (torch.rand(windows.shape, **kwargs).to(windows) < spike_prob).to(windows.dtype)
        spikes = torch.randn(windows.shape, **kwargs).to(windows) * spike_scale
        noisy = noisy + mask * spikes
    return noisy


def build_windows(sequence: torch.Tensor, spec: WindowSpec) -> torch.Tensor:
    """Slice a ``(T, J)`` trajectory into ``(N, length, J)`` windows.

    Window ``i`` ends at timestep ``i + span - 1``, so window ``i`` uses only
    information available at its final timestep -- no lookahead.
    """
    if sequence.ndim != 2:
        raise ValueError(f"sequence must be (T, J), got {tuple(sequence.shape)}")
    steps = sequence.shape[0]
    if steps < spec.span:
        return sequence.new_zeros((0, spec.length, sequence.shape[1]))
    offsets = torch.arange(spec.length) * spec.stride
    starts = torch.arange(steps - spec.span + 1)
    return sequence[starts[:, None] + offsets[None, :]]


def latent_gap(command_embedding: torch.Tensor, execution_embedding: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """Cosine distance between unit-normalized embeddings, in ``[0, 2]``.

    Normalizing first makes the gap a measure of *direction* in latent space, so
    it does not simply grow with the magnitude of the motion -- a vigorous
    motion tracked well should score better than a gentle motion tracked badly.
    """
    if command_embedding.shape != execution_embedding.shape:
        raise ValueError(
            f"embeddings must align: {tuple(command_embedding.shape)} vs "
            f"{tuple(execution_embedding.shape)}"
        )
    a = command_embedding / (command_embedding.norm(dim=-1, keepdim=True) + eps)
    b = execution_embedding / (execution_embedding.norm(dim=-1, keepdim=True) + eps)
    return 1.0 - (a * b).sum(dim=-1)


def raw_mismatch(command_windows: torch.Tensor, execution_windows: torch.Tensor) -> torch.Tensor:
    """Per-window Frobenius norm of raw joint-space error, the LUCID ablation."""
    if command_windows.shape != execution_windows.shape:
        raise ValueError("command and execution windows must align")
    return (command_windows - execution_windows).flatten(1).norm(dim=-1)


def gap_series(
    encoder: TemporalVAE,
    commanded: torch.Tensor,
    executed: torch.Tensor,
    spec: WindowSpec = WindowSpec(),
) -> dict[str, torch.Tensor]:
    """Latent and raw gap series for one episode.

    Returns both so an audit can ask whether the latent representation adds
    anything over raw joint error, which is exactly LUCID v1's own ablation.
    """
    if commanded.shape != executed.shape:
        raise ValueError(
            f"commanded {tuple(commanded.shape)} and executed {tuple(executed.shape)} must align"
        )
    command_windows = build_windows(commanded, spec)
    execution_windows = build_windows(executed, spec)
    if command_windows.shape[0] == 0:
        empty = commanded.new_zeros((0,))
        return {"latent": empty, "raw": empty, "warmup_steps": spec.span - 1}
    return {
        "latent": latent_gap(encoder.embed(command_windows), encoder.embed(execution_windows)),
        "raw": raw_mismatch(command_windows, execution_windows),
        "warmup_steps": spec.span - 1,
    }


@dataclass
class GapSummary:
    """Summary of a gap series, matching LUCID's scheduling statistics."""

    median: float
    p90: float
    mean: float
    variance: float
    slope: float
    num_windows: int

    def to_dict(self) -> dict[str, float]:
        return {
            "gap_median": self.median,
            "gap_p90": self.p90,
            "gap_mean": self.mean,
            "gap_variance": self.variance,
            "gap_slope": self.slope,
            "gap_num_windows": float(self.num_windows),
        }


def summarize_gap(series: torch.Tensor, quantile: float = 0.9) -> GapSummary:
    """Median, high quantile, variance, and temporal slope of a gap series.

    The high quantile is LUCID's scheduling statistic -- it emphasizes
    near-failure behaviour that a mean would average away. The slope is included
    because a gap that is *growing* within an episode means something different
    from a steady one of the same magnitude.

    Note the quantile's blind spot: degradation occupying less than ``1 -
    quantile`` of an episode sits inside the tail and does not move ``p90`` at
    all, while the mean does. A brief excursion is therefore invisible to a
    p90-driven curriculum by construction. Both statistics are reported so an
    audit can tell which regime a context is in.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    values = series.detach().to(torch.float64).reshape(-1)
    if values.numel() == 0:
        return GapSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    if values.numel() == 1:
        v = float(values[0])
        return GapSummary(v, v, v, 0.0, 0.0, 1)

    steps = torch.arange(values.numel(), dtype=torch.float64)
    centred_steps = steps - steps.mean()
    denominator = float(centred_steps.pow(2).sum())
    slope = (
        float((centred_steps * (values - values.mean())).sum() / denominator)
        if denominator > 0 else 0.0
    )
    return GapSummary(
        median=float(values.median()),
        p90=float(torch.quantile(values, quantile)),
        mean=float(values.mean()),
        variance=float(values.var(unbiased=False)),
        slope=slope,
        num_windows=int(values.numel()),
    )


def fill_after_termination(
    series: torch.Tensor, terminated_at: int | None, horizon: int
) -> torch.Tensor:
    """Extend a gap series to ``horizon`` with the terminal value.

    A policy that falls at step 10 has no gap for steps 11..100. Truncating
    instead of filling would make an early fall look like a short, tidy episode
    with a small average gap -- rewarding exactly the failure being measured.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    values = series.detach().reshape(-1)
    if terminated_at is not None:
        values = values[: max(0, terminated_at)]
    if values.numel() >= horizon:
        return values[:horizon]
    padding = values.new_full((horizon - values.numel(),), TERMINATED_GAP)
    return torch.cat([values, padding])


def train_encoder(
    windows: torch.Tensor,
    num_joints: int,
    spec: WindowSpec = WindowSpec(),
    latent_dim: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    beta: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log_every: int = 0,
) -> tuple[TemporalVAE, list[dict[str, Any]]]:
    """Pre-train the temporal VAE, then freeze it.

    Self-supervised: the clean window is the reconstruction target and the
    corrupted window is the input, so no labels are needed and the encoder
    learns to discard transients. The returned model is in eval mode with
    gradients disabled -- it is an instrument, and an instrument that drifts
    during the experiment measures nothing.
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must be (N, T, J), got {tuple(windows.shape)}")
    if windows.shape[0] == 0:
        raise ValueError("no training windows supplied")

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = TemporalVAE(num_joints, spec.length, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data = windows.to(device)
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        order = torch.randperm(data.shape[0], generator=generator).to(device)
        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        batches = 0
        for start in range(0, data.shape[0], batch_size):
            clean = data[order[start : start + batch_size]]
            noisy = corrupt_windows(clean, generator=None)
            reconstruction, mu, logvar = model(noisy)
            loss, parts = elbo_loss(reconstruction, clean, mu, logvar, beta=beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss)
            totals["recon"] += parts["recon"]
            totals["kl"] += parts["kl"]
            batches += 1
        record = {"epoch": epoch, **{k: v / max(batches, 1) for k, v in totals.items()}}
        history.append(record)
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:3d}  loss {record['loss']:.5f}  recon {record['recon']:.5f}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, history


def encoder_fingerprint(model: TemporalVAE) -> str:
    """Hash of the frozen weights, recorded with every gap measurement.

    Gaps from different encoders are not comparable, so each measurement carries
    the identity of the instrument that produced it.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
