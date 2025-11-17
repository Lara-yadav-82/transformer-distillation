# Transformer Distillation Training

Train a T5-based transformer model using knowledge distillation techniques.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Your Data

Place your data files in the `dataset/` folder:
- `final_train_10k.jsonl` - Training dataset
- `final_val_10k.jsonl` - Validation dataset

### 3. Run Training

```bash
python code/train_distillation_advanced.py
```

## Data Format

Your JSONL files should follow this format:

```json
{"text": "Input text", "summary": "Target summary"}
{"text": "Another input", "summary": "Another summary"}
```

## System Requirements

**Minimum (CPU):**
- RAM: 16GB+
- Storage: 5GB
- Training time: 10-20 hours

**Recommended (GPU):**
- GPU: NVIDIA with 8GB+ VRAM
- RAM: 16GB+
- CUDA: 11.8+
- Training time: 2-4 hours

## Output

Training results will be saved in the `result/` folder.

## Troubleshooting

- **Out of Memory**: Reduce batch size in the training script
- **CUDA errors**: Check PyTorch GPU installation
- **Import errors**: Run `pip install -r requirements.txt` again

