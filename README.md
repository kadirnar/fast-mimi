![fast-mimi](assets/fast-mimi-banner.png)

# Fast-Mimi

Fast-Mimi is an independent PyTorch inference runtime for the frozen
[Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural audio codec. Production
code under `src/fast_mimi` does not import or depend on `transformers`.
The original checkpoint is loaded directly with `huggingface-hub` and
`safetensors`.

The default portable path is pure PyTorch. The guarded RTX 5070 Ti/SM120 path
adds Inductor graphs, CUDA Graph replay, Triton memory kernels, cuDNN frontend
plan selection, and a native CUTLASS decoder-tail kernel. Unsupported shapes,
versions, devices, or toolchains fail closed to the portable implementation.

The declared model identity is locked to:

- Model: `kyutai/mimi`
- Revision: `89091b3e466eb6a9d11e537bf26b144f194978f7`
- Weights SHA-256: `bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e`
- Parameter fingerprint: `3feaa6168b191ffdebfd8f695b963f72c8d847a3966f7cc3283af6b38d437bb4`
- Parameter count: `79,308,609`

## 100 saniyelik optimizasyon sonuçları

Ölçümler RTX 5070 Ti (SM120), 24 kHz mono, sekiz codebook ve tam 100 saniye
(2.400.000 örnek) içindir. İlk çağrı; derleme, plan seçimi ve autotune süresiyle
birlikte ölçüm dışında bırakılmıştır. Son yayımlanabilir API sonucu 50+50
dönüşümlü örnek ve 10.000 bootstrap tekrarına dayanır. Farklı araştırma
oturumlarındaki küçük taban kaymalarını açık tutmak için bazı satırlarda hem
uçtan uca değer hem de o deneyin mevcut yola göre artımlı oranı yazılmıştır.
Üretim paketi yalnız bağımsız PyTorch, Triton, cuDNN, CUDA Graph ve CUDA/CUTLASS
kodunu içerir; üretim ağacında `transformers` importu yoktur.

| Teknik / çalışma yolu | Hassasiyet / backend | 100 sn uçtan uca medyan | Ölçülen hızlanma | Doğruluk kanıtı ve karar |
|---|---|---:|---:|---|
| Sıfırdan bağımsız PyTorch | FP32 | **135.667 ms** | **1.0000x** | Dondurulmuş referans; 79.308.609 parametre ve sekiz codebook |
| Önceki Fast-Mimi sürümü | FP32 | 87.091 ms | 1.5578x | Yerine daha hızlı sürüm geçti |
| Inductor + CUDA Graph + yerel pencere + seçili Triton/cuDNN paketi | FP32 | 65.848 ms | 2.0603x | Kabul edilen ara sürüm |
| Kalite güvenli ayrık bottleneck; RVQ seçimi Inductor dışında | FP32 | 62.235 ms | 2.1800x | Kabul; 20/20 uzun ses, kodlar birebir |
| Native CUTLASS decoder katman 11, 22 SM120 varyantı | FP16 giriş, FP32 birikim/çıkış | 62.235 ms paketine dahil | Önceki 65.848 ms pakete göre 1.0645x | Kabul; kalite sıralamalı cuDNN plan kurtarmasıyla |
| Decoder-9 cuDNN + WMMA füzyonu ve native final-post | FP16 dal, FP32 residual/çıkış | 60.878 ms | 2.2299x | Kabul; 20/20 kalite kapısı geçti |
| Paketli QKV GEMM + bit-düzeyi eş RoPE | FP32 + Triton | Tek başına 61.014 ms oturumu | 2.2249x; tek başına artış <%1 | Tek başına reddedildi, yalnız birleşik adayda kabul edildi |
| Encoder `max-autotune-no-cudagraphs` | FP32 Inductor | Birleşik 59.636 ms sonucuna dahil | Tek başına <%1 | Birebir çıktı; yalnız birleşik adayda kabul edildi |
| Sabit-pointer encoder suffix CUDA Graph | FP32 | Birleşik 59.636 ms sonucuna dahil | Encoder bölümü hızlandı, tek başına <%1 uçtan uca | Birebir çıktı; birleşik adayda kabul edildi |
| Sabit-pointer bottleneck CUDA Graph | FP32 | Birleşik 59.636 ms sonucuna dahil | Tek başına uçtan uca ≈1.00x | Birebir kod/decoded tensor; birleşik adayda kabul edildi |
| Decoder prefix-2 autotune | FP32 Inductor | Birleşik 59.636 ms sonucuna dahil | Tek başına <%1 | Birebir çıktı; birleşik adayda kabul edildi |
| Sabit-pointer decoder CUDA Graph | FP32 | Birleşik 59.636 ms sonucuna dahil | Tek başına <%1 | Birebir çıktı; birleşik adayda kabul edildi |
| Decoder-12/final WMMA, ilk Tile=64/8 warp | FP16 noktasal, FP32 residual/final | 60.570 ms | Mevcut yola göre 1.0052x | Kalite geçti; tek başına pratik <%1 kazanç nedeniyle reddedildi |
| Decoder-12/final WMMA, seçilen Tile=48/4 warp | FP16 noktasal, FP32 residual/final | 59.956 ms | Mevcut yola göre 1.0144x; tail 1.2295x | Kabul; tüm tile varyantlarıyla bit-düzeyi eş, 20/20 geçti |
| Tüm yeni seçeneklerin birleşik fast-kernel adaptörü | FP32 + Triton/cuDNN/CUDA/CUTLASS | **59.636 ms** | **2.2744x; %95 GA 2.2713x–2.2755x** | Kabul; 20/20, kod farkı 0, tolerans ihlali 0 |
| Yayımlanan bağımsız Fast-Mimi API; kullanıcıya ait output clone dahil | FP32 + Triton/cuDNN/CUDA/CUTLASS | **59.881 ms** | **2.2656x; %95 GA 2.2642x–2.2685x** | **Kabul; 20/20, dört native çekirdek yüklendi, runtime fallback oluşmadı** |
| Güncel dondurulmuş 100 sn kapısı; fast-kernel adaptörü | FP32 + Triton/cuDNN/CUDA/CUTLASS | **133.561 → 58.916 ms** | **2.2669x; %95 GA 2.2638x–2.2852x** | **Kabul; 20/20, kod farkı 0, tolerans ihlali 0, en kötü oran 0.887200** |
| Güncel profil — convolutional encoder | FP32/cuDNN/Inductor/CUDA Graph | 25.947 ms | Uçtan uca sürenin %44,00’ü | En büyük kalan darboğaz; sonraki optimizasyon hedefi |
| Güncel profil — encoder attention/MLP | FP32, paketli QKV + CUDA Graph | 8.397 ms | Uçtan uca sürenin %14,24’ü | İkinci grup; düşük hassasiyet denemeleri kaliteyi geçmedi |
| Güncel profil — kalite güvenli RVQ bottleneck | FP32 + sabit CUDA Graph | 1.248 ms | Uçtan uca sürenin %2,12’si | Küçük pay; kod sınırları nedeniyle FP32 seçim korunuyor |
| Güncel profil — decoder attention/MLP | FP32, paketli QKV + CUDA Graph | 8.517 ms | Uçtan uca sürenin %14,44’ü | Üçüncü grup; yaklaşık attention yolları kaliteyi geçmedi |
| Güncel profil — convolutional decoder | Karışık kalite güvenli FP16/FP32 | 14.528 ms | Uçtan uca sürenin %24,64’ü | İkinci en büyük tek aşama; decoder-9/11/12 native füzyonları etkin |
| Monolitik Inductor bottleneck; RVQ `cdist`/`argmin` dahil | FP32 | 61.851 ms | 2.1934x | Reddedildi; seed 1103 iki nearest-code sınırını geçti |
| Kalite sıralaması olmadan en hızlı cuDNN planları | FP32 | 61.836 ms | 2.1941x | Reddedildi; 31 ses toleransı ihlali |
| FlashAttention-4 CuTe DSL | FP16 | ≈59.62 ms | Mevcut yola göre 1.0179x | Reddedildi; 3–13 kod farkı ve çok sayıda ses ihlali |
| FlashAttention-4 CuTe DSL | BF16 | ≈59.55 ms | Mevcut yola göre 1.0190x | Reddedildi; FP16’dan daha kötü kalite |
| GemLite transformer | FP8 | 56.520 ms | Mevcut yola göre 1.0736x; referansa göre ≈2.400x | Reddedildi; 349–423 kod farkı ve ≈2,2 milyon ihlal |
| HQQ + GemLite transformer | INT4 ağırlık | ≈56.35 ms | Mevcut yola göre 1.0768x; referansa göre ≈2.408x | Reddedildi; 1.363–1.729 kod farkı ve ≈2,35 milyon ihlal |
| TileLang tüm attention/MLP transformer GEMM’leri | FP32 giriş, TF32 donanım yolu | 55.910 ms | Mevcut yola göre 1.0861x; referansa göre ≈2.427x | Reddedildi; 70–95 kod farkı ve 1,12–1,33 milyon ihlal |
| cuDNN attention, iki attention stack | FP16 | 58.624 ms | 2.3141x | Reddedildi; kod ve dalga kalitesi başarısız |
| cuDNN attention, iki attention stack | BF16 | 58.653 ms | 2.3130x | Reddedildi; kod ve dalga kalitesi başarısız |
| Düzeltilmiş birikimli cuDNN attention | FP32 | 59.452 ms | 2.2820x | Reddedildi; kod/ses ihlalleri kaldı |
| Yalnız decoder cuDNN attention | FP32 | 60.427 ms | 2.2451x | Reddedildi; yüzlerce ses ihlali |
| Global TF32 attention/MLP GEMM | TF32 | 57.435 ms | 2.3621x | Reddedildi; 15–30 kod ve 358 bin–434 bin ses ihlali |
| Yalnız decoder TF32 attention/MLP GEMM | TF32 | 59.648 ms | 2.2745x | Reddedildi; kodlar eş fakat 10 bin–17 bin ses ihlali |
| Seçici TF32 QKV/output/FC1/FC2 grupları | TF32 | Kalite nedeniyle geçerli uçtan uca sonuç yok | Artımlı 1.0023x–1.0273x | Reddedildi; her grup dalga toleransını aştı |
| CuTe DSL attention katman 7, altı GEMM | FP16 | 61.490 ms | 2.2063x; artımlı 1.0062x | Reddedildi; 13.187 ihlal ve <%1 kazanç |
| CuTe DSL attention katman 7, altı GEMM | BF16 | 61.492 ms | 2.2062x; artımlı 1.0065x | Reddedildi; 335.855 ihlal |
| Üç terimli telafili CuTe DSL | FP16 + FP32 telafi | 62.033 ms | 2.1871x; artımlı 0.9974x | Reddedildi; daha yavaş ve 1.974 ihlal |
| Sekiz decoder transformer katmanını `torch.compile` | FP32 | ≈59.82 ms | Mevcut yola göre 1.0145x | Reddedildi; 421–641 ses ihlali |
| Seçici compiled encoder | FP32 | ≈59.82 ms | Mevcut yola göre 1.0144x | Reddedildi; kod ve ses kapısı başarısız |
| Tam forward CUDA Graph | FP32 | 60.803 ms | 60.773 ms mevcut yola göre 0.9995x | Reddedildi; birebir fakat daha yavaş |
| Decoder residual blok 6, iki convolution | FP16 | 60.875 ms | 2.2286x | Reddedildi; ses ihlali |
| Decoder residual blok 6, ilk convolution | FP16 | 61.425 ms | 2.2087x | Reddedildi; ses ihlali |
| Decoder residual blok 6, son convolution | FP16 | 61.950 ms | 2.1900x; artımlı 1.00015x | Tek başına <%1 olduğu için reddedildi |
| Decoder residual blok 6, ilk convolution | BF16 | 61.490 ms | 2.2063x | Reddedildi; 189.135–338.842 ihlal |
| FP16 kanal kurtarma taraması | Karışık FP16/FP32 | Geçerli uçtan uca sonuç yok | Blok bileşeni ≈2.0x | Reddedildi; 128 kanalın en az 112’si FP32 kalınca kazanç silindi |
| QKV füzyonu + Triton scaled residual | FP32 | 62.828 ms | 2.1592x; artımlı 0.9948x | Reddedildi; daha yavaş |
| Scaled-residual özel Triton kernel | FP32 | Geçerli uçtan uca kazanç yok | PyTorch 0.0209 ms, Triton ≈0.0309 ms | Reddedildi; bileşen daha yavaş |
| RVQ paketli projection | FP32 | ≈60.7 ms | Encode 1.021x; uçtan uca 0.997x–1.000x | Reddedildi; birebir fakat pratik kazanç yok |
| Semantic/acoustic ilk `cdist` batching | FP32 | ≈60.7 ms | Bottleneck 1.1615x; uçtan uca 0.997x–1.000x | Reddedildi; uçtan uca ilerleme yok |
| cuDNN deconv exact-plan taraması | FP32 | Mevcut yol ile aynı | Deconv-8 plan 1/9 ≈1.877–1.878 ms | Reddedildi; exact plan kazanç sağlamadı |
| cuDNN deconv hızlı yaklaşık planlar | FP32 | Kalite nedeniyle geçerli değil | Daha hızlı deconv-2/5 | Reddedildi; 1–31 ses ihlali |
| Encoder NHWC/channels-last zinciri | FP32 | 1.388,750 ms | 0.0977x | Reddedildi; daha yavaş ve yanlış |
| Özel Triton encoder, decoder, attention, RoPE, RVQ ve normlar | FP32 | 555.001 ms | 0.2444x | Reddedildi; 16 kod ve 492.355 ses ihlali |
| Encoder suffix/stage-4 yaklaşık füzyon | FP32 | Kalite nedeniyle geçerli değil | Encoder 26.631→22.122 ms, 1.204x | Reddedildi; 22 kod farkı; exact kurtarma yalnız 1.036x encoder kazancı |
| Encoder stage-7 füzyon | FP32 | Kalite nedeniyle geçerli değil | Stage 4.731→4.576 ms | Reddedildi; yedi kod farkı ve encoder düzeyinde kazanç yok |
| Encoder stage-10 füzyon | FP32 | Kalite nedeniyle geçerli değil | Stage 2.488→2.420 ms | Reddedildi; altı kod farkı ve encoder düzeyinde kazanç yok |
| Exact segmented encoder kurtarma | FP32 | Mevcut yoldan daha yavaş | Yaklaşık hızlı yolun doğruluğunu geri getirdi | Reddedildi; kurtarma maliyeti kazancı sildi |
| Alternatif yerel-attention geometrileri (32/256–250/250) | FP32 | Kalite nedeniyle geçerli değil | Güvenilir kazanç yok | Reddedildi; seed 1103 başarısız, çoğu varyant daha yavaş |
| TileLang Q projection | FP32 | Bileşen doğruluğu başarısız | 0.0473 ms; bileşen 1.574x | Reddedildi; maksimum hata 0.02073 |
| TileLang FC1 | FP32 | Bileşen doğruluğu başarısız | 0.1399 ms; bileşen 1.750x | Reddedildi; maksimum hata 0.01163 |
| ModelOpt encoder | FP8 | Bileşen doğruluğu başarısız | 81.796 ms; compiled FP32’ye göre 0.375x | Reddedildi; daha yavaş ve 3.015 kod farkı |
| HQQ + GemLite | INT8 ağırlık | Bileşen doğruluğu başarısız | 0.2214 ms; bileşen 0.919x | Reddedildi; daha yavaş, maksimum hata 0.1867 |
| HQQ + GemLite | INT4 ağırlık | Bileşen doğruluğu başarısız | 0.2370 ms; bileşen 0.859x | Reddedildi; daha yavaş, maksimum hata 2.016 |
| TensorRT encoder en iyi yapılandırma | FP32 birikim | Kod kapısı başarısız | Encoder 20.901 ms; 1.467x | Reddedildi; tüm seed’lerde kod farkı |
| TensorRT decoder en iyi yapılandırma | FP32 | Bileşen kapısı başarısız | Decoder 154.457 ms; 0.111x | Reddedildi; yaklaşık 9x daha yavaş ve kalite başarısız |
| TF32 ve 3xTF32 GEMM varyantları | Karışık | Bileşen doğruluğu başarısız | GEMM’ler daha hızlı | Reddedildi; kod farkları |
| Native cuBLAS BF16x9 telafili GEMM | BF16 parçalar, FP32 toplam | ≈mevcut yol | ≈0.99x | Ertelendi; SM120’de yerleşik yol FP32’ye düştü, Tensor Core hızlanması oluşmadı |
| BF16x3 + decoder-12 Tile=64/5, düzeltilmiş clone kapısı | BF16x3 + FP32 residual | 58.605 → 58.241 ms | 1.0063x; %95 GA 1.0018x–1.0151x | Reddedildi; seed başına 963–1.465 ses ihlali |
| Seçici BF16x3 QKV/FC1 taraması, sekiz decoder katmanı | BF16x3 + FP32 | Geçerli uçtan uca sonuç yok | 24 seçenek tarandı | Reddedildi; her tekil katman/operatör 342–1.204 ses ihlali üretti |
| Exact NHWC encoder stage-4/down-6 zinciri | FP32 cuDNN | 8.414 → 792.618 ms (aşama) | 0.0106x | Reddedildi; bit-düzeyi eş fakat seçilen exact plan çok yavaş |
| Exact NHWC residual-4 hibriti | FP32 cuDNN/Inductor | 8.412 → 9.174 ms (aşama) | 0.9170x | Reddedildi; bit-düzeyi eş fakat layout dönüşü kazancı sildi |
| CUTLASS encoder down-6, altı SM120 implicit-conv ayarı | FP32 | 4.198 → 4.754 ms (bileşen) | 0.8832x | Reddedildi; daha yavaş ve maksimum fark 3.81e-5 |
| Native cuBLAS FP32 transformer GEMM taraması | FP32 | Bileşen bazında | QKV 1.0196x; O 1.0476x; FC1 1.0083x; FC2 1.0152x | Bit-düzeyi eş; tekil/Amdahl kazancı pratik eşiğin altında |
| cuBLASLt exact transformer algoritmaları | FP32 | 58.925 → 58.810 ms | 1.0020x; %95 GA 0.9939x–1.0071x | Reddedildi; kalite tam geçti fakat uçtan uca kazancı istatistiksel ve pratik değil |
| cuBLASLt hızlı FC1 algoritması | FP32, farklı reduction planı | 58.925 → ≈58.54 ms | 1.0066x; %95 GA 1.0030x–1.0144x | Reddedildi; 5 kod farkı ve bir seed’de 178.915 ses ihlali |
| cuDNN residual-6 füzyonu | FP32 graph | 58.955 → 58.655 ms | 1.0051x; %95 GA 0.9968x–1.0108x; blok 1.175x | Reddedildi; 3 ses ihlali ve uçtan uca GA 1x’i kapsıyor |
| cuDNN residual-6 exact-numerics plan filtresi | FP32 SIMT | Çalıştırılabilir engine yok | — | SM120 cuDNN backend’inde uygun exact fused engine için ertelendi |
| cuBLASLt residual-6 exact epilogue | FP32 | 2.114 → 2.376 ms (blok) | 0.8897x | Reddedildi; daha yavaş ve seçilen matmul planı kaliteyi geçmedi |
| CUTLASS SIMT encoder pointwise zinciri | FP32 | 8.023 → 6.432 ms (prefix) | 1.2473x | Reddedildi; hızlı fakat final encoder’da 13 kod farkı oluşturdu |
| Segmented cuDNN attention | FP32 | Bileşen kapısı başarısız | 0.270 ms; bileşen 0.937x | Reddedildi; aynı hatayla daha yavaş |
| cuDNN multi-MMA residual-branch graph | FP32 | Engine üretilemedi | Fixed-pointer fallback decoder 17.2471→17.2045 ms, 1.0025x | SM120 cuDNN backend güncellemesine ertelendi |
| FP32 giriş + FP16/BF16 ağırlık cuDNN graph | Karışık | Çalıştırılabilir engine yok | — | SM120 cuDNN graph desteğine ertelendi |
| ModelOpt/TensorRT-LLM FP8 export | FP8 | Export başarısız | — | CUDA 12 uyumlu hedefe ertelendi; eager FP8 kaliteyi zaten geçemedi |
| Nsight Compute donanım sayaçları | SM120 | `ERR_NVGPUCTRPERM` | — | Yönetici izni olan profil hedefine ertelendi |

Kabul edilen yol bütün 79.308.609 parametreyi ve sekiz codebook’u korur.
20 dondurulmuş 100 saniyelik girdide kod farkı ve ses toleransı ihlali sıfırdır;
en kötü tolerans oranı `0.887200`, maksimum mutlak hata `0.000197440` olmuştur.
Dalga çıktısı bit-düzeyi eş olarak değil, dondurulmuş `atol=2e-4,
rtol=1e-4` sözleşmesi içinde kalite eşdeğeri olarak tanımlanır.

## Install

Portable PyTorch runtime:

```bash
pip install "fast-mimi @ git+https://github.com/kadirnar/fast-mimi.git"
```

RTX 5070 Ti/SM120 optimized runtime:

```bash
pip install "fast-mimi[optimized] @ git+https://github.com/kadirnar/fast-mimi.git"
```

The optimized extra installs the uv-distributed CUDA 13 `nvcc` and CCCL
packages; TileLang supplies the CUTLASS header tree. If Triton, cuDNN frontend,
TileLang, `nvcc`, CUDA 13, SM120, or the validated shape contract is
unavailable, Fast-Mimi uses the portable path. Set
`FAST_MIMI_DISABLE_OPTIMIZED_LONG=1` to force that fallback.

## Quick start

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

Short clips, arbitrary supported shapes, streaming calls, CPU execution, and
non-SM120 GPUs continue to use the independent PyTorch implementation.

## Reproducing the accepted contract

The validated optimized path requires PyTorch `2.13.0+cu130`, Triton `3.7.1`,
cuDNN frontend `1.27.0`, TileLang `0.1.13`, CUDA runtime 13.0, the uv CUDA 13.3
compiler/CCCL packages, Linux, SM120,
`torch.backends.cuda.matmul.allow_tf32 == False`, and
`torch.backends.cudnn.allow_tf32 == True`. These are dispatch guards, not
silent global-setting mutations.

The CUTLASS implementation follows NVIDIA's
[SM120 functionality](https://docs.nvidia.com/cutlass/latest/overview.html),
[CuTe DSL](https://docs.nvidia.com/cutlass/latest/index.html), and
[autotuning](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/autotuning_gemm.html)
documentation.

## License

Fast-Mimi is licensed under [Apache-2.0](LICENSE). The `kyutai/mimi` weights
are distributed separately under CC-BY-4.0. Mimi was introduced in Kyutai's
[Moshi repository](https://github.com/kyutai-labs/moshi) and
[Moshi paper](https://arxiv.org/abs/2410.00037).
