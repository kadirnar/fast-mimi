![fast-mimi](assets/fast-mimi-banner.png)

# Fast-Mimi

Fast-Mimi, dondurulmuş [Kyutai Mimi](https://huggingface.co/kyutai/mimi)
checkpoint'i için sıfırdan yazılmış bağımsız bir PyTorch inference runtime'ıdır.
`src/fast_mimi` altında Transformers importu veya bağımlılığı yoktur; checkpoint
doğrudan `huggingface-hub` ve `safetensors` ile yüklenir.

Taşınabilir yol saf PyTorch kullanır. RTX 5070 Ti/SM120 için doğrulanmış yol;
Inductor, CUDA Graph, Triton, cuDNN frontend ve native CUDA/CUTLASS çekirdeklerini
birleştirir. Desteklenmeyen cihaz, şekil veya araç zincirinde güvenli biçimde saf
PyTorch yoluna döner. Reddedilmiş deneysel implementasyonlar üretim paketine dahil
edilmemiştir.

Model kimliği kod içinde şu değerlere kilitlidir:

- Model: `kyutai/mimi`
- Revision: `89091b3e466eb6a9d11e537bf26b144f194978f7`
- Weights SHA-256: `bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e`
- Parameter fingerprint: `3feaa6168b191ffdebfd8f695b963f72c8d847a3966f7cc3283af6b38d437bb4`
- Parameter count: `79,308,609`

## 100 saniyelik uçtan uca sonuçlar

RTX 5070 Ti (SM120), 24 kHz mono, sekiz codebook ve tam 2.400.000 örnek
ölçülmüştür. Derleme/autotune içeren ilk çağrı ölçüm dışıdır. Yalnızca kabul
edilmiş ve üretim kodunda bulunan optimizasyonlar gösterilir.

| Kabul edilen yol | Hassasiyet / backend | 100 sn uçtan uca medyan | Hızlanma | Doğruluk sonucu |
|---|---|---:|---:|---|
| Bağımsız saf PyTorch referansı | FP32 | 135.667 ms | 1.0000x | Dondurulmuş mimari ve checkpoint |
| Inductor + CUDA Graph + Triton/cuDNN temel paketi | FP32 | 65.848 ms | 2.0603x | Geçti |
| Kalite güvenli RVQ + cuDNN plan kurtarması | FP32 | 62.235 ms | 2.1800x | 20/20, kodlar birebir |
| Native CUTLASS decoder-11 | FP16 giriş, FP32 birikim/çıkış | 62.235 ms paketine dahil | Önceki pakete göre 1.0645x | Geçti |
| cuDNN + WMMA decoder-9 ve native final-post | FP16 dal, FP32 residual/çıkış | 60.878 ms | 2.2299x | 20/20 geçti |
| Seçilen WMMA decoder-12/final | FP16 noktasal, FP32 residual/final | 59.956 ms | 2.2628x | 20/20 geçti |
| Paketli QKV, bit-eş RoPE, encoder/bottleneck/decoder sabit-pointer graph ve autotune birleşimi | FP32 + Triton/cuDNN/CUDA/CUTLASS | 59.636 ms | 2.2744x | 20/20; kod farkı 0 |
| Yayımlanan bağımsız Fast-Mimi API | FP32 + Triton/cuDNN/CUDA/CUTLASS | 59.881 ms | 2.2656x | 20/20; fallback yok |
| Güncel dondurulmuş paired kapı | FP32 + Triton/cuDNN/CUDA/CUTLASS | **133.561 → 58.916 ms** | **2.2669x; %95 GA 2.2638x–2.2852x** | **20/20; kod farkı 0; ihlal 0** |

Son kapı 50 dönüşümlü ölçüm çifti ve 10.000 bootstrap tekrarıyla çalıştırıldı.
En kötü tolerans oranı `0.887200`, maksimum mutlak ses farkı `0.000197440` oldu.
Dalga kalitesi dondurulmuş `atol=2e-4, rtol=1e-4` sözleşmesiyle korunur.

## Kurulum

Saf PyTorch runtime:

```bash
pip install "fast-mimi @ git+https://github.com/kadirnar/fast-mimi.git"
```

RTX 5070 Ti/SM120 optimize runtime:

```bash
pip install "fast-mimi[optimized] @ git+https://github.com/kadirnar/fast-mimi.git"
```

## Kullanım

```python
import torch

from fast_mimi import MimiModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MimiModel.from_pretrained(device=device)

audio = torch.randn(1, 1, 2_400_000, device=device) * 0.05
padding_mask = torch.ones_like(audio, dtype=torch.bool)

with torch.inference_mode():
    output = model(audio, padding_mask=padding_mask, num_quantizers=8)

print(output.audio_codes.shape)
print(output.audio_values.shape)
```

Kısa sesler, farklı desteklenen şekiller, streaming, CPU ve SM120 dışındaki
GPU'lar bağımsız saf PyTorch yolunu kullanır. Optimize sözleşme PyTorch
`2.13.0+cu130`, Triton `3.7.1`, cuDNN frontend `1.27.0`, TileLang `0.1.13`,
CUDA runtime 13.0, CUDA 13.3 compiler/CCCL ve Linux SM120 ile doğrulanmıştır.

## Lisans

Fast-Mimi [Apache-2.0](LICENSE) lisanslıdır. `kyutai/mimi` ağırlıkları ayrıca
CC-BY-4.0 ile dağıtılır.
