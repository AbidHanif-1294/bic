"""
Simulasi jaringan neuron spiking (LIF) → sinyal EEG sintetik.

Alur:
    Poisson input (rangsangan)
        ↓
    Layer 1: Thalamic relay neurons (LIF)
        ↓
    Layer 2: Cortical pyramidal neurons (LIF, eksitatori + inhibitori)
        ↓
    Layer 3: Proyeksi ke elektroda (weighted sum + lowpass + noise)
        ↓
    Sinyal EEG (n_channels, n_times) @ fs=256 Hz

Referensi konsep:
- LIF: dv/dt = -(v - v_rest)/tau_m + I/C
- Emisi spike ketika v > v_thresh, lalu reset ke v_reset
- EEG ≈ jumlah post-synaptic potential dari populasi neuron piramidal
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ============================================================
# Parameter neuron (satuan disederhanakan, tidak dalam SI)
# ============================================================
@dataclass
class LIFParams:
    tau_m: float = 20.0        # konstanta waktu membran (ms)
    v_rest: float = -70.0      # potensial istirahat (mV)
    v_reset: float = -75.0     # setelah spike
    v_thresh: float = -50.0    # ambang spike
    t_ref: float = 2.0         # periode refraktori (ms)
    r_in: float = 10.0         # resistansi input (MOhm)


# ============================================================
# Populasi neuron LIF
# ============================================================
class LIFPopulation:
    """
    Populasi neuron LIF dengan konektivitas acak (sparse).
    """

    def __init__(
        self,
        n_neurons: int,
        params: LIFParams,
        excitatory_ratio: float = 0.8,
        connection_prob: float = 0.1,
        seed: int = 0,
    ):
        self.n = n_neurons
        self.p = params
        rng = np.random.default_rng(seed)

        # Neuron tipe: 80% eksitatori (E), 20% inhibitori (I)
        self.is_excitatory = rng.random(n_neurons) < excitatory_ratio
        # Bobot sinaptik: E positif, I negatif
        self.synaptic_sign = np.where(self.is_excitatory, 1.0, -4.0)

        # Matriks konektivitas sparse
        mask = rng.random((n_neurons, n_neurons)) < connection_prob
        np.fill_diagonal(mask, False)
        self.W = (mask * rng.uniform(0.5, 1.5, size=mask.shape)) * self.synaptic_sign[None, :]

        # State
        self.v = np.full(n_neurons, params.v_rest, dtype=np.float64)
        self.refractory_until = np.zeros(n_neurons, dtype=np.float64)

    def step(self, I_ext: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Satu langkah simulasi Euler.
        I_ext: input eksternal per neuron (shape: n_neurons)
        t: waktu sekarang (ms)
        dt: step (ms)
        return: spike vector (1 kalau spike, 0 kalau tidak)
        """
        p = self.p
        # Neuron di periode refraktori tidak update
        active = t >= self.refractory_until

        # dV/dt = -(V - V_rest)/tau + R*I/tau
        dV = (-(self.v - p.v_rest) + p.r_in * I_ext) / p.tau_m
        self.v[active] += dt * dV[active]

        # Emisi spike
        spikes = np.zeros(self.n, dtype=np.float64)
        fired = active & (self.v >= p.v_thresh)
        spikes[fired] = 1.0
        self.v[fired] = p.v_reset
        self.refractory_until[fired] = t + p.t_ref

        return spikes


# ============================================================
# Jaringan 2-lapis: Thalamus → Korteks
# ============================================================
class CorticalNetwork:
    """
    Dua populasi:
    - Thalamic (n_thal) menerima Poisson input eksternal
    - Cortical (n_cort) menerima input dari thalamus + recurren sendiri
    Output: firing rate cortical (proxy untuk EEG)
    """

    def __init__(
        self,
        n_thal: int = 100,
        n_cort: int = 200,
        seed: int = 0,
    ):
        self.thal = LIFPopulation(n_thal, LIFParams(), connection_prob=0.05, seed=seed)
        self.cort = LIFPopulation(n_cort, LIFParams(), connection_prob=0.1, seed=seed + 1)

        rng = np.random.default_rng(seed + 2)
        # Proyeksi thalamus → korteks (feedforward)
        self.W_thal_cort = rng.uniform(0.2, 0.8, size=(n_cort, n_thal))
        # Threshold koneksi: hanya 30% yang connect
        self.W_thal_cort *= (rng.random((n_cort, n_thal)) < 0.3)

        self.n_thal = n_thal
        self.n_cort = n_cort

    def step(self, I_thal_ext: np.ndarray, t: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        # 1) Thalamus
        thal_spikes = self.thal.step(I_thal_ext, t, dt)

        # 2) Input ke korteks = proyeksi thalamus + recurrent
        I_cort_from_thal = self.W_thal_cort @ thal_spikes
        I_cort_rec = self.cort.W @ self.cort_spikes_last if hasattr(self, "cort_spikes_last") else 0.0
        I_cort = I_cort_from_thal + I_cort_rec

        # 3) Korteks
        cort_spikes = self.cort.step(I_cort, t, dt)
        self.cort_spikes_last = cort_spikes
        self.thal_spikes_last = thal_spikes

        return thal_spikes, cort_spikes


# ============================================================
# Konversi spike → EEG
# ============================================================
class EEGProjector:
    """
    Proyeksi firing rate populasi kortikal → sinyal EEG multi-channel.
    Menggunakan prinsip: EEG ≈ jumlah PSP dari neuron piramidal
    yang berorientasi sama, di-lowpass oleh volume conduction.

    Output per channel = konvolusi spike train dengan kernel PSP (alpha function),
    kemudian di-mix via matriks proyeksi acak.
    """

    def __init__(
        self,
        n_cort: int,
        n_channels: int = 8,
        fs: float = 256.0,
        tau_psp: float = 10.0,  # ms
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed + 100)
        # Setiap channel "melihat" subset neuron kortikal dengan bobot acak
        self.P = rng.normal(0, 1, size=(n_channels, n_cort))
        self.P /= (np.linalg.norm(self.P, axis=1, keepdims=True) + 1e-9)

        self.fs = fs
        self.tau_psp = tau_psp

        # Kernel PSP: alpha function h(t) = (t/tau) * exp(1 - t/tau)
        t_kernel = np.arange(0, 5 * tau_psp, 1)  # ms, sampel 1 ms
        self.kernel = (t_kernel / tau_psp) * np.exp(1 - t_kernel / tau_psp)
        self.kernel /= self.kernel.sum()

    def project(self, spike_rate_trace: np.ndarray) -> np.ndarray:
        """
        spike_rate_trace: (n_times_ms, n_cort) — firing rate per ms
        return: (n_channels, n_times_ms) sinyal EEG
        """
        # Konvolusi tiap neuron dengan kernel PSP
        eeg_per_neuron = np.apply_along_axis(
            lambda x: np.convolve(x, self.kernel, mode="same"),
            axis=0,
            arr=spike_rate_trace,
        )
        # Mix ke channel
        eeg = self.P @ eeg_per_neuron.T  # (n_channels, n_times_ms)
        # Downsample dari 1000 Hz → fs (misal 256 Hz)
        downsample_factor = int(1000 / self.fs)
        eeg = eeg[:, ::downsample_factor]
        return eeg


# ============================================================
# End-to-end: generate EEG sintetik dari simulasi neuron
# ============================================================
def simulate_brain_eeg(
    duration_sec: float = 60.0,
    n_channels: int = 8,
    fs: float = 256.0,
    n_thal: int = 100,
    n_cort: int = 200,
    stim_freq: float = 10.0,       # Hz — osilasi dominan (alpha)
    stim_amplitude: float = 3.0,   # kekuatan rangsangan
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate sinyal EEG multi-channel dari simulasi jaringan neuron.

    Returns
    -------
    eeg : np.ndarray, shape (n_channels, n_times)
        Sinyal EEG sintetik dalam satuan mikrovolt (µV).
    """
    rng = np.random.default_rng(seed)

    net = CorticalNetwork(n_thal=n_thal, n_cort=n_cort, seed=seed)
    projector = EEGProjector(n_cort, n_channels=n_channels, fs=fs, seed=seed)

    # Simulasi pada resolusi 1 ms (1000 Hz), lalu di-downsample
    dt = 1.0  # ms
    n_steps_ms = int(duration_sec * 1000)

    # Buffer untuk spike train (0/1 per ms)
    thal_trace = np.zeros((n_steps_ms, n_thal), dtype=np.float32)
    cort_trace = np.zeros((n_steps_ms, n_cort), dtype=np.float32)

    if verbose:
        print(f"[sim] durasi {duration_sec}s, dt={dt}ms, steps={n_steps_ms}")

    for i in range(n_steps_ms):
        t = i * dt

        # Rangsangan eksternal ke thalamus: osilasi + noise
        oscillation = stim_amplitude * (1 + np.sin(2 * np.pi * stim_freq * t / 1000.0))
        I_ext = oscillation + rng.normal(0, 0.5, n_thal)

        thal_sp, cort_sp = net.step(I_ext, t, dt)
        thal_trace[i] = thal_sp
        cort_trace[i] = cort_sp

        if verbose and i % 5000 == 0:
            print(f"  t = {t/1000:.1f}s | thal spikes = {thal_sp.sum():.0f} | cort spikes = {cort_sp.sum():.0f}")

    # Spike train → firing rate (smoothing window 5 ms)
    def smooth_rate(trace, k=5):
        kernel = np.ones(k) / k
        return np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"),
            axis=0, arr=trace,
        )

    if verbose:
        print("[sim] menghitung firing rate...")
    thal_rate = smooth_rate(thal_trace) * 1000.0  # Hz
    cort_rate = smooth_rate(cort_trace) * 1000.0

    if verbose:
        print("[sim] proyeksi ke EEG...")
    eeg = projector.project(cort_rate)  # (n_channels, n_times @ fs)

    # Skala ke µV (biar mirip EEG asli)
    eeg = eeg * 15.0
    eeg += rng.normal(0, 2.0, eeg.shape)  # noise sensor

    if verbose:
        print(f"[sim] selesai. EEG shape: {eeg.shape}")
        print(f"      range: [{eeg.min():.2f}, {eeg.max():.2f}] µV")

    return eeg.astype(np.float32)
