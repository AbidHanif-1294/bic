"""
Dynamic Neural System untuk memodelkan dinamika EEG.

Arsitektur:
- Encoder (GRU/RNN-style) memetakan window EEG -> latent state z_t
- Latent dynamics: dz/dt = f(z, x) dengan time constant tau
- Decoder merekonstruksi sinyal EEG dari latent trajectory
- Predictor memprediksi window berikutnya (simulasi aktivitas)

Model menghasilkan "internal representation" pola dinamika subject.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNeuralSystem(nn.Module):
    """
    Model dinamika neural kontinu dengan discrete-time integration.

    Equations:
        z_t = tanh(W_in @ x_t + W_rec @ z_{t-1} + b)
        z_{t+1} = z_t + dt/tau * (-z_t + f(W_rec z_t + W_in x_t + b))
        x_hat_t = Decoder(z_t)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        tau: float = 0.1,
        dt: float = 0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.tau = tau
        self.dt = dt

        # Encoder: EEG window -> feature (per timestep)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Recurrent dynamic core
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Continuous-time dynamics (recurrent weight untuk Euler integration)
        self.W_rec = nn.Linear(latent_dim, latent_dim, bias=True)

        # Decoder: latent -> reconstructed EEG window
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Predictor: latent_t -> latent_{t+1} untuk simulasi
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) -> latent: (B, T, latent_dim)
        """
        # transpose ke (B, C, T) untuk Conv1d
        h = self.encoder(x.transpose(1, 2))   # (B, H, T)
        h = h.transpose(1, 2)                  # (B, T, H)
        z, _ = self.rnn(h)                     # (B, T, latent_dim)
        return z

    def continuous_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """
        Satu step Euler integration dari dynamics kontinu:
            dz/dt = (-z + tanh(W_rec z + b)) / tau
        """
        drive = torch.tanh(self.W_rec(z))
        dz = (-z + drive) / self.tau
        return z + self.dt * dz

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, T, latent_dim) -> x_hat: (B, T, C)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, T, C)
        Returns dict dengan:
            - reconstruction: (B, T, C)
            - latent: (B, T, latent_dim)
            - latent_next: (B, T, latent_dim)  (prediksi state berikutnya)
            - dynamics_z: (B, T, latent_dim)   (state setelah continuous dynamics)
        """
        z = self.encode(x)                          # (B, T, L)
        z_dyn = self.continuous_dynamics(z)         # (B, T, L)
        recon = self.decode(z_dyn)                  # (B, T, C)
        z_next = self.predictor(z_dyn)              # (B, T, L)

        return {
            "reconstruction": recon,
            "latent": z,
            "latent_dynamics": z_dyn,
            "latent_next": z_next,
        }

    @torch.no_grad()
    def simulate(
        self,
        x_seed: torch.Tensor,
        n_steps: int,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Simulasi aktivitas EEG ke depan dari seed window.
        x_seed: (B, T_seed, C)
        return: (B, n_steps, C) -- prediksi window EEG berikutnya
        """
        self.eval()
        z = self.encode(x_seed)                      # (B, T_seed, L)
        z_t = z[:, -1, :]                            # (B, L) state terakhir

        outputs = []
        for _ in range(n_steps):
            z_t = self.continuous_dynamics(z_t.unsqueeze(1)).squeeze(1)
            z_t = self.predictor(z_t)
            if temperature > 0:
                z_t = z_t + temperature * torch.randn_like(z_t)

            # Decode state ke vektor EEG (rata-rata window)
            x_hat = self.decoder(z_t.unsqueeze(1)).squeeze(1)  # (B, C)
            outputs.append(x_hat)

        return torch.stack(outputs, dim=1)           # (B, n_steps, C)


class DynamicLoss(nn.Module):
    """Combined loss: reconstruction + prediction + latent smoothness."""

    def __init__(
        self,
        alpha_recon: float = 1.0,
        alpha_pred: float = 0.5,
        alpha_smooth: float = 0.1,
        alpha_latent_reg: float = 0.01,
    ):
        super().__init__()
        self.a_r = alpha_recon
        self.a_p = alpha_pred
        self.a_s = alpha_smooth
        self.a_l = alpha_latent_reg

    def forward(self, outputs: dict, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        recon = outputs["reconstruction"]            # (B, T, C)
        z_dyn = outputs["latent_dynamics"]           # (B, T, L)
        z_next = outputs["latent_next"]              # (B, T, L)

        loss_recon = F.mse_loss(recon, x)

        # Prediction: latent_t+1 ≈ z_dyn_{t+1}
        if z_dyn.size(1) > 1:
            loss_pred = F.mse_loss(z_next[:, :-1, :], z_dyn[:, 1:, :].detach())
        else:
            loss_pred = torch.tensor(0.0, device=x.device)

        # Smoothness: state tidak melompat drastis
        if z_dyn.size(1) > 1:
            diff = z_dyn[:, 1:, :] - z_dyn[:, :-1, :]
            loss_smooth = diff.pow(2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=x.device)

        # Regularisasi latent (mencegah explosion)
        loss_latent = z_dyn.pow(2).mean()

        total = (
            self.a_r * loss_recon
            + self.a_p * loss_pred
            + self.a_s * loss_smooth
            + self.a_l * loss_latent
        )

        return total, {
            "recon": loss_recon.item(),
            "pred": loss_pred.item() if isinstance(loss_pred, torch.Tensor) else 0.0,
            "smooth": loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else 0.0,
            "latent": loss_latent.item(),
            "total": total.item(),
        }
