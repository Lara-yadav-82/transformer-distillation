# Text Summarization Transformer Training

Train a 124M parameter T5-based transformer model for text summarization.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install PyTorch (GPU version - recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

**OR** if you have CPU only:
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt
```

### 2. Prepare Your Data

Make sure you have these files in the same directory:
- `train_sample.jsonl` - Training data
- `val_sample.jsonl` - Validation data

### 3. Run Training

```bash
python train_summarization.py
```

## 📊 Model Architecture

- **Type**: T5 (Text-to-Text Transfer Transformer)
- **Parameters**: ~124 Million
- **Encoder Layers**: 12
- **Decoder Layers**: 12
- **Hidden Dimension**: 768
- **Attention Heads**: 12

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 4 |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 16 |
| Learning Rate | 5e-5 |
| Warmup Steps | 500 |
| Max Source Length | 512 tokens |
| Max Target Length | 128 tokens |

## 📁 Output Files

After training, you'll find:

```
model_outputs/
├── best_model.pt              # PyTorch checkpoint
├── best_model_hf/             # HuggingFace format model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── training_history.png       # Loss curves plot
```

## 💻 System Requirements

### Minimum (CPU Training):
- RAM: 16GB+
- Storage: 5GB
- Training Time: ~10-20 hours

### Recommended (GPU Training):
- GPU: NVIDIA with 8GB+ VRAM (RTX 3060, T4, or better)
- RAM: 16GB+
- CUDA: 11.8+
- Training Time: ~2-4 hours

## 🎯 Using the Trained Model

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

# Load model
model = T5ForConditionalGeneration.from_pretrained('./model_outputs/best_model_hf')
tokenizer = AutoTokenizer.from_pretrained('./model_outputs/best_model_hf')

# Summarize text
text = "Your long text here..."
inputs = tokenizer("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=128, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

## 📈 Evaluation Metrics

The script automatically calculates:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

## 🛠️ Customization

Edit these variables in `train_summarization.py`:

```python
# Line 32-40
EPOCHS = 10                      # Number of training epochs
BATCH_SIZE = 4                   # Batch size (reduce if OOM)
GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation
LEARNING_RATE = 5e-5             # Learning rate
MAX_SOURCE_LENGTH = 512          # Max input length
MAX_TARGET_LENGTH = 128          # Max summary length
```

## ⚠️ Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
BATCH_SIZE = 2  # or even 1
```

### Slow Training
```python
# Reduce max lengths
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 64
```

### CUDA Not Available
- Make sure you installed PyTorch with CUDA support
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

## 📝 Data Format

Your JSONL files should have this format:

```json
{"text": "Long text to summarize...", "summary": "Short summary"}
{"text": "Another long text...", "summary": "Another summary"}
```

## 🤝 Support

Common issues:
1. **Import errors**: Run `pip install -r requirements.txt`
2. **CUDA errors**: Update GPU drivers or use CPU version
3. **Data loading errors**: Check JSONL file format

## 📄 License

This training script is provided as-is for educational purposes.
