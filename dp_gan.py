# dp_gan.py
# Differentially private GAN for preprocessed tabular data.
#
# Privacy design (the standard DP-GAN construction):
#   * Only the discriminator ever sees real records. It is trained with DP-SGD
#     (per-sample gradient clipping + Gaussian noise) through Opacus, with
#     Poisson-sampled batches so the privacy accountant's assumptions hold.
#   * The generator never touches real data; it only receives gradients from the
#     (already private) discriminator. By the post-processing property of
#     differential privacy, the generator and every row it samples inherit the
#     discriminator's (epsilon, delta) guarantee.
#
# Each real record and one fake record are paired in the discriminator step, so a
# record's clipped per-sample gradient covers both terms of its loss.

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset


class _Generator(nn.Module):
    def __init__(self, noise_dim: int, hidden: tuple, out_dim: int):
        super().__init__()
        layers, dim = [], noise_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.LeakyReLU(0.2)]
            dim = h
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple):
        super().__init__()
        layers, dim = [], in_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DPGAN:
    """GAN whose discriminator is trained with DP-SGD (Opacus)."""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        max_grad_norm: float = 1.0,
        epochs: int = 300,
        batch_size: int = 500,
        noise_dim: int = 64,
        generator_dim: tuple = (128, 128),
        discriminator_dim: tuple = (128, 128),
        lr: float = 3e-3,
        gumbel_tau: float = 0.2,
        verbose: bool = True,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1")
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.lr = lr
        self.gumbel_tau = gumbel_tau
        self.verbose = verbose
        self.device = torch.device("cpu")  # small MLPs; CPU keeps Opacus behaviour predictable

        self.columns: List[str] = []
        self._numeric_idx: List[int] = []
        self._groups: List[List[int]] = []
        self.generator: Optional[_Generator] = None
        self.privacy_report: Dict = {}

    # ------------------------------------------------------------------ layout
    def _build_layout(self, columns: List[str], column_groups: Optional[Dict[str, List[str]]],
                      discrete_columns: List[str]) -> None:
        self.columns = list(columns)
        position = {c: i for i, c in enumerate(self.columns)}
        grouped = set()
        self._groups = []
        for names in (column_groups or {}).values():
            idx = [position[n] for n in names if n in position]
            if idx:
                self._groups.append(idx)
                grouped.update(idx)
        # Discrete columns without a known group are treated as independent binary flags.
        for name in discrete_columns or []:
            i = position.get(name)
            if i is not None and i not in grouped:
                self._groups.append([i])
                grouped.add(i)
        self._numeric_idx = [i for i in range(len(self.columns)) if i not in grouped]

    def _activate(self, raw: torch.Tensor, hard: bool) -> torch.Tensor:
        """Apply output activations: linear for numeric, (Gumbel-)softmax for one-hot groups."""
        out = torch.empty_like(raw)
        if self._numeric_idx:
            out[:, self._numeric_idx] = raw[:, self._numeric_idx]
        for idx in self._groups:
            logits = raw[:, idx]
            if len(idx) == 1:
                probs = torch.sigmoid(logits)
                out[:, idx] = (probs > torch.rand_like(probs)).float() if hard else probs
            else:
                out[:, idx] = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=hard)
        return out

    # ---------------------------------------------------------------- training
    def fit(self, data: pd.DataFrame, discrete_columns: Optional[List[str]] = None,
            column_groups: Optional[Dict[str, List[str]]] = None) -> Dict:
        self._build_layout(list(data.columns), column_groups, discrete_columns or [])
        n_rows, dim = data.shape
        if n_rows < 10:
            raise ValueError("Differentially private training needs at least 10 rows.")

        # Cap the sampling rate at about 1/3: privacy amplification by subsampling needs
        # it well below 1, and it gives the generator enough steps on small datasets.
        batch_size = int(max(1, min(self.batch_size, max(16, n_rows // 3), n_rows)))
        x = torch.tensor(data.to_numpy(dtype=np.float32), dtype=torch.float32)
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

        generator = _Generator(self.noise_dim, self.generator_dim, dim).to(self.device)
        discriminator = _Discriminator(dim, self.discriminator_dim).to(self.device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        engine = PrivacyEngine(accountant="rdp")
        discriminator, opt_d, loader = engine.make_private_with_epsilon(
            module=discriminator,
            optimizer=opt_d,
            data_loader=loader,  # replaced by a Poisson-sampling loader
            epochs=self.epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )

        bce = nn.BCEWithLogitsLoss()
        steps = 0
        for epoch in range(self.epochs):
            loss_d_val = loss_g_val = float("nan")
            for (real,) in loader:
                n = real.shape[0]
                if n == 0:
                    continue
                real = real.to(self.device)

                # Discriminator step (DP-SGD). Real record i is paired with fake record i.
                discriminator.train()
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake = self._activate(generator(torch.randn(n, self.noise_dim, device=self.device)), hard=False)
                loss_d = (bce(discriminator(real), torch.ones(n, 1, device=self.device))
                          + bce(discriminator(fake), torch.zeros(n, 1, device=self.device)))
                loss_d.backward()
                opt_d.step()
                steps += 1

                # Generator step: no real data involved, so no privacy cost.
                discriminator.disable_hooks()
                opt_g.zero_grad(set_to_none=True)
                fake = self._activate(generator(torch.randn(batch_size, self.noise_dim, device=self.device)), hard=False)
                loss_g = bce(discriminator(fake), torch.ones(batch_size, 1, device=self.device))
                loss_g.backward()
                opt_g.step()
                discriminator.enable_hooks()
                opt_d.zero_grad(set_to_none=True)  # drop discriminator grads from the generator pass

                loss_d_val, loss_g_val = loss_d.item(), loss_g.item()

            if self.verbose and (epoch % 25 == 0 or epoch == self.epochs - 1):
                logging.info(f"DP-GAN epoch {epoch + 1}/{self.epochs}  loss_D={loss_d_val:.3f}  "
                             f"loss_G={loss_g_val:.3f}  ε so far={engine.get_epsilon(self.delta):.3f}")

        eps_spent, best_alpha = engine.accountant.get_privacy_spent(delta=self.delta)
        self.generator = generator.eval()
        self.privacy_report = {
            "epsilon": float(eps_spent),
            "delta": self.delta,
            "alpha": float(best_alpha),
            "noise_multiplier": float(opt_d.noise_multiplier),
            "max_grad_norm": self.max_grad_norm,
            "sample_rate": batch_size / n_rows,
            "expected_batch_size": batch_size,
            "steps": steps,
        }
        return self.privacy_report

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, num_rows: int) -> pd.DataFrame:
        if self.generator is None:
            raise ValueError("Model has not been trained yet.")
        chunks, remaining = [], num_rows
        while remaining > 0:
            n = min(remaining, 4096)
            raw = self.generator(torch.randn(n, self.noise_dim, device=self.device))
            chunks.append(self._activate(raw, hard=True).cpu().numpy())
            remaining -= n
        values = np.vstack(chunks)
        if self._numeric_idx:
            # Numeric columns are standardised; keep samples within a plausible range.
            values[:, self._numeric_idx] = np.clip(values[:, self._numeric_idx], -5, 5)
        return pd.DataFrame(values, columns=self.columns)
