# EEG-Driven Dynamic Neural Network — Prototype Penelitian

Sistem eksperimental: EEG Subject A → preprocessing → dynamic neural network →
representasi internal pola dinamika → rekonstruksi & simulasi aktivitas.

## Konsep

**Model A** adalah instance Dynamic Neural System yang dilatih pada data EEG
Subject A. Model mempelajari representasi laten dari dinamika temporal EEG
menggunakan:

1. **Encoder Conv1D** — ekstraksi fitur lokal per timestep.
2. **GRU recurrent core** — pemodelan dependensi temporal.
3. **Continuous-time dynamics** — Euler integration: `dz/dt = (-z + tanh(W z)) / τ`.
4. **Decoder** — rekonstruksi sinyal EEG.
5. **Predictor** — prediksi state laten berikutnya untuk simulasi.

## Instalasi

```bash
pip install -r requirements.txt
