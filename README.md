# ImageAudioGen

Projeto completo de inteligência artificial generativa que implementa dois modelos de deep learning: geração de imagens com Diffusion Models e regeneração de áudio com Autoencoders.

## 📋 Descrição do Projeto

Este projeto avalia capacidades técnicas em IA aplicada, demonstrando:
- Domínio de frameworks de deep learning (PyTorch)
- Conhecimento de modelos generativos (Diffusion Models e Autoencoders)
- Boas práticas de engenharia de machine learning
- Pré-processamento e avaliação de dados

## 🏗️ Arquitetura do Projeto

```
ImageAudioGen/
├── image_gen.py        # Geração de imagens com Diffusion Model
├── audio_gen.py        # Regeneração de áudio com Autoencoder
├── README.md           # Este arquivo
└── challenge.txt       # Descrição dos requisitos
```

---

## 📸 Parte 1: Geração de Imagens (`image_gen.py`)

### O que faz?
Implementa um **Diffusion Model** completo capaz de gerar imagens sintéticas de dígitos manuscritos (MNIST) a partir de ruído Gaussiano aleatório.

### Componentes Principais

#### 1. **PositionalEncoding**
- Codifica informação temporal (timesteps) da difusão
- Usa funções seno/cosseno para criar embeddings posicionais
- Permite que a rede entenda em qual estágio do processo de difusão está

#### 2. **DiffusionModel (U-Net)**
- **Entrada**: Imagem com ruído + timestep
- **Saída**: Predição do ruído Gaussiano adicionado
- **Arquitetura**: U-Net simplificada com:
  - Encoder: 2 blocos de convolução + maxpooling (reduz dimensionalidade)
  - Decoder: 2 blocos de deconvolução (restaura tamanho original)
  - Skip connections: Concatenam features do encoder com decoder

#### 3. **DiffusionTrainer**
- **Forward Diffusion**: Adiciona ruído progressivamente à imagem (1000 timesteps)
- **Reverse Diffusion**: Remove ruído iterativamente para gerar novas imagens
- **Perda**: MSE entre ruído predito e ruído real

#### 4. **Métricas e Avaliação**
- **FID (Fréchet Inception Distance)**: Mede qualidade/diversidade das imagens geradas
- Comparação visual entre épocas
- Histórico de perda de treinamento

### Como Usar

**Treinar o modelo:**
```bash
python image_gen.py --mode train \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --device cuda
```

**Gerar amostras com modelo treinado:**
```bash
python image_gen.py --mode infer \
    --num_samples 16 \
    --checkpoint models/diffusion_model.pt \
    --device cuda
```

### Argumentos Disponíveis

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--mode` | str | train | 'train' para treinar ou 'infer' para gerar |
| `--epochs` | int | 20 | Número de épocas de treinamento |
| `--batch_size` | int | 64 | Tamanho do batch |
| `--learning_rate` | float | 1e-3 | Taxa de aprendizado |
| `--checkpoint` | str | models/diffusion_model.pt | Caminho do modelo salvo |
| `--num_samples` | int | 16 | Quantidade de imagens a gerar |
| `--device` | str | cuda/cpu | CPU ou GPU |

### Outputs Gerados

```
results/
├── training_loss.png           # Gráfico de perda vs época
├── epoch_comparisons/
│   ├── samples_epoch_001.png   # Amostras na época 1
│   ├── samples_epoch_005.png   # Amostras na época 5
│   ├── samples_epoch_010.png   # Amostras na época 10
│   ├── samples_epoch_020.png   # Amostras na época 20
│   └── metrics_comparison.png  # FID vs Loss vs Época
└── final_samples.png           # Amostras finais geradas

models/
└── diffusion_model.pt          # Pesos do modelo treinado
```

### Resultados Esperados

- **Qualidade**: Imagens cada vez melhores conforme aumenta o treinamento
- **Diversidade**: FID aumenta (maior desvio padrão = mais diversidade)
- **Perda**: Diminui exponencialmente nas primeiras épocas

---

## 🔊 Parte 2: Regeneração de Áudio (`audio_gen.py`)

### O que faz?
Implementa um **Autoencoder** para reconstruir stems de áudio a partir de representações no domínio tempo-frequência (Mel-Spectrogram).

### Componentes Principais

#### 1. **AudioPreprocessor**
- Converte áudio em **Mel-Spectrogram** (análise em frequência)
- **Mel-Spectrogram**: Representação que imita como o ouvido humano percebe som
  - Frequências: Representadas em escala logarítmica (Mel)
  - Eixo Y: 128 Mel bins (padrão)
  - Eixo X: Frames de tempo
- Usa **Griffin-Lim** para reconstruir áudio a partir do Mel-Spectrogram
- Normaliza dados para treinamento

#### 2. **SyntheticMUSDBDataset**
- Simula dataset MUSDB18 (um padrão em separação de áudio)
- Gera áudio sintético com:
  - Múltiplas frequências harmônicas (440Hz, 880Hz, 1320Hz, 1760Hz)
  - Amplitudes variáveis
  - Ruído Gaussiano
- Redimensiona para tamanho fixo (256 frames temporais)

#### 3. **AudioAutoencoder**
- **Entrada**: Mel-Spectrogram [128 mels × 256 timesteps]
- **Processo**:
  1. Encoder (4 camadas): Comprime para espaço latente (64 dimensões)
  2. Bottleneck: Representação comprimida
  3. Decoder (4 camadas): Reconstrói Mel-Spectrogram original
- **Saída**: Mel-Spectrogram reconstruído [128 × 256]

#### 4. **AudioTrainer**
- **Função de Perda**: MSE entre Mel-Spectrograms original e reconstruído
- **Otimizador**: Adam com learning rate scheduler
- **Gradient Clipping**: Evita explosão de gradientes

#### 5. **Métricas de Qualidade**

| Métrica | Descrição | Intervalo |
|---------|-----------|-----------|
| **MSE** | Erro quadrático médio pixel a pixel | 0-∞ (menor=melhor) |
| **MAE** | Erro absoluto médio | 0-∞ (menor=melhor) |
| **Cosine Similarity** | Similaridade entre espectros | 0-1 (maior=melhor) |
| **PESQ Proxy** | Aproximação de qualidade perceptual | 0-1 (maior=melhor) |

### Como Usar

**Treinar o modelo:**
```bash
python audio_gen.py --mode train \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_samples 100 \
    --n_mels 128 \
    --latent_dim 64 \
    --device cuda
```

**Reconstruir áudio com modelo treinado:**
```bash
python audio_gen.py --mode infer \
    --checkpoint models/audio_autoencoder.pt \
    --device cuda
```

### Argumentos Disponíveis

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--mode` | str | train | 'train' para treinar ou 'infer' para reconstruir |
| `--epochs` | int | 30 | Número de épocas de treinamento |
| `--batch_size` | int | 32 | Tamanho do batch |
| `--learning_rate` | float | 1e-3 | Taxa de aprendizado |
| `--num_samples` | int | 100 | Quantidade de amostras do dataset |
| `--checkpoint` | str | models/audio_autoencoder.pt | Caminho do modelo salvo |
| `--n_mels` | int | 128 | Número de Mel bins |
| `--latent_dim` | int | 64 | Dimensão do espaço latente |
| `--device` | str | cuda/cpu | CPU ou GPU |

### Outputs Gerados

```
results/
├── training_curves.png                # Perda + Métricas vs Época
├── spectrogram_comparison.png         # Original vs Reconstruído
├── inference_comparison.png           # Amostras de teste
├── audio_reconstructed_0.wav          # Áudio reconstruído #0
└── audio_reconstructed_1.wav          # Áudio reconstruído #1

models/
└── audio_autoencoder.pt               # Pesos do modelo treinado
```

### Fluxo de Processamento

```
Áudio original (16kHz, 5 segundos)
         ↓
   Mel-Spectrogram
   [1 × 128 × 256]
         ↓
    ENCODER
   (4 camadas Conv1d)
         ↓
   Espaço Latente
   [1 × 64]
         ↓
    DECODER
   (4 camadas ConvTranspose1d)
         ↓
   Mel-Spectrogram Reconstruído
   [1 × 128 × 256]
         ↓
Griffin-Lim Inverse
         ↓
Áudio Reconstruído (16kHz)
```

### Resultados Esperados

- **MSE**: Decresce durante treinamento (começa ~0.5, fim ~0.05)
- **Similaridade**: Aumenta (começa ~0.5, fim ~0.95)
- **Qualidade Perceptual**: Áudio reconstruído cada vez mais fiel ao original

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install torch torchaudio torchvision torchmetrics numpy matplotlib soundfile tqdm
```

### Execução Completa (Imagem + Áudio)

```bash
# Passo 1: Treinar modelo de geração de imagens
python image_gen.py --mode train --epochs 20 --batch_size 64

# Passo 2: Gerar novas imagens
python image_gen.py --mode infer --num_samples 16

# Passo 3: Treinar modelo de regeneração de áudio
python audio_gen.py --mode train --epochs 30 --batch_size 32

# Passo 4: Reconstruir áudio
python audio_gen.py --mode infer
```

---

## 📊 Métricas e Resultados

### Diffusion Model (Imagens)
- **Métrica Principal**: FID (Fréchet Inception Distance)
- **Visualização**: Comparação de amostras entre épocas
- **Loss**: MSE entre ruído predito e real

### Autoencoder (Áudio)
- **Métricas**: MSE, MAE, Cosine Similarity, PESQ Proxy
- **Visualização**: Espectrogramas original vs reconstruído
- **Análise**: Gráficos de convergência

---

## 🎯 Destaques Técnicos

✅ **Modelos Generativos**: Diffusion Models (SOTA em geração de imagens)  
✅ **Modelos Autoencoders**: Compressão e reconstrução eficiente  
✅ **Pré-processamento**: Mel-Spectrogram para áudio, Normalização para ambos  
✅ **Métricas Avançadas**: FID, Cosine Similarity, PESQ Proxy  
✅ **Best Practices**: Checkpointing, Learning Rate Scheduling, Gradient Clipping  
✅ **CLI Intuitiva**: Argumentos configuráveis para fácil experimentação  
✅ **Visualização**: Gráficos comparativos e análise de qualidade  

---

## 📝 Frameworks Utilizados

- **PyTorch**: Deep learning framework principal
- **Torchaudio**: Processamento de áudio
- **Torchvision**: Transformações de imagem
- **Matplotlib**: Visualização
- **Soundfile**: Exportação de áudio
- **Tqdm**: Barras de progresso

---

## 📚 Referências

- Ho et al. (2020): Denoising Diffusion Probabilistic Models (DDPM)
- Kingma & Welling (2013): Auto-Encoding Variational Bayes
- Mel-Frequency Cepstral Coefficients (MFCC) - Padrão em processamento de áudio

---

## ✨ Próximos Passos

- Implementar VAE (Variational Autoencoder) para áudio
- Adicionar GAN para geração de imagens
- Integração com dados MUSDB18 reais
- API REST para inferência
- Web interface com Streamlit