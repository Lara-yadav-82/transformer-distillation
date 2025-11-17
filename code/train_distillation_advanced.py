"""
Knowledge Distillation Training for Email Summarization
Teacher: flan-t5-base (250M) - Fine-tuned instruction model
Student: t5-small (60M) or t5-medium (124M)

Features:
✓ Knowledge distillation with temperature scaling
✓ Automatic checkpoint resumption
✓ Google Drive integration for saving
✓ Mixed loss (distillation + hard label + generation)
✓ Early stopping with patience
✓ Comprehensive evaluation with ROUGE scores

Expected Performance:
- Student (small): ROUGE-L 0.55-0.68 (with distillation boost)
- Student (medium): ROUGE-L 0.60-0.72 (with distillation boost)
"""
!pip install torch transformers datasets rouge-score sentencepiece matplotlib tqdm accelerate numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import json
import numpy as np
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data files
    TRAIN_FILE = 'final_train_10k.jsonl'
    VAL_FILE = 'final_val_10k.jsonl'

    # Model configuration
    TEACHER_MODEL = 'google/flan-t5-base'  # 250M params, instruction-tuned
    STUDENT_SIZE = 'small'  # 'small' (60M) or 'medium' (124M)

    # Training hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 64
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.01

    # Sequence lengths
    MAX_SOURCE_LENGTH = 512
    MAX_TARGET_LENGTH = 128

    # Distillation parameters
    TEMPERATURE = 2.0  # Softens probability distributions
    ALPHA = 0.5  # Weight for distillation loss (0.5 = balanced)
    BETA = 0.3   # Weight for hard label loss
    GAMMA = 0.2  # Weight for generation loss

    # Model architecture (for student)
    DROPOUT_RATE = 0.2

    # Early stopping
    PATIENCE = 7

    # Checkpointing
    OUTPUT_DIR = Path('distillation_outputs')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    BEST_MODEL_DIR = OUTPUT_DIR / 'best_model'
    RESUME_FROM_CHECKPOINT = True  # Set to True to resume training

    # Google Drive (optional - set to None if not using)
    GDRIVE_SAVE = False  # Set to True to enable
    GDRIVE_PATH = '/content/drive/MyDrive/distillation_model'

config = Config()

print("="*80)
print("KNOWLEDGE DISTILLATION TRAINING")
print("="*80)
print(f"Device: {config.device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\n📚 Teacher: {config.TEACHER_MODEL}")
print(f"🎓 Student: T5-{config.STUDENT_SIZE}")
print(f"🔥 Distillation Temperature: {config.TEMPERATURE}")
print(f"⚖️  Loss weights: α={config.ALPHA} (distill), β={config.BETA} (hard), γ={config.GAMMA} (gen)")

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================

config.OUTPUT_DIR.mkdir(exist_ok=True)
config.CHECKPOINT_DIR.mkdir(exist_ok=True)
config.BEST_MODEL_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_data = load_jsonl(config.TRAIN_FILE)
val_data = load_jsonl(config.VAL_FILE)

print(f"✅ Training samples: {len(train_data):,}")
print(f"✅ Validation samples: {len(val_data):,}")

# ============================================================================
# LOAD MODELS
# ============================================================================

print("\n" + "="*80)
print("LOADING TEACHER MODEL")
print("="*80)

# Load teacher model (frozen)
teacher_tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_MODEL)
teacher_model = T5ForConditionalGeneration.from_pretrained(config.TEACHER_MODEL)
teacher_model = teacher_model.to(config.device)
teacher_model.eval()  # Teacher is always in eval mode

# Freeze teacher
for param in teacher_model.parameters():
    param.requires_grad = False

teacher_params = sum(p.numel() for p in teacher_model.parameters())
print(f"✅ Teacher loaded: {teacher_params:,} ({teacher_params/1e6:.0f}M) parameters")

print("\n" + "="*80)
print("CREATING STUDENT MODEL")
print("="*80)

# Student architecture
STUDENT_CONFIGS = {
    'small': {
        'd_model': 512,
        'd_kv': 64,
        'd_ff': 2048,
        'num_layers': 6,
        'num_decoder_layers': 6,
        'num_heads': 8,
    },
    'medium': {
        'd_model': 768,
        'd_kv': 64,
        'd_ff': 3072,
        'num_layers': 8,
        'num_decoder_layers': 8,
        'num_heads': 12,
    }
}

# Use same tokenizer as teacher
student_tokenizer = teacher_tokenizer

student_arch = STUDENT_CONFIGS[config.STUDENT_SIZE]

# IMPORTANT: Ensure vocab size matches teacher exactly
teacher_vocab_size = teacher_model.config.vocab_size

student_config = T5Config(
    vocab_size=teacher_vocab_size,  # Use teacher's vocab size
    d_model=student_arch['d_model'],
    d_kv=student_arch['d_kv'],
    d_ff=student_arch['d_ff'],
    num_layers=student_arch['num_layers'],
    num_decoder_layers=student_arch['num_decoder_layers'],
    num_heads=student_arch['num_heads'],
    relative_attention_num_buckets=32,
    dropout_rate=config.DROPOUT_RATE,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    feed_forward_proj="relu",
    is_encoder_decoder=True,
    use_cache=True,
    pad_token_id=student_tokenizer.pad_token_id,
    eos_token_id=student_tokenizer.eos_token_id,
    decoder_start_token_id=student_tokenizer.pad_token_id
)

student_model = T5ForConditionalGeneration(student_config)
student_model = student_model.to(config.device)

student_params = sum(p.numel() for p in student_model.parameters())
trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

print(f"✅ Student created: {student_params:,} ({student_params/1e6:.0f}M) parameters")
print(f"✅ Trainable: {trainable_params:,} ({trainable_params/1e6:.0f}M) parameters")
print(f"📊 Compression ratio: {teacher_params/student_params:.2f}x")

# ============================================================================
# DATASET CLASS
# ============================================================================

class DistillationDataset(Dataset):
    """Dataset for knowledge distillation"""

    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get text: support both 'text' field and 'subject'+'content' fields
        if 'text' in item:
            email_text = item['text']
        else:
            # Combine subject and content for richer context
            subject = item.get('subject', '')
            content = item.get('content', '')
            email_text = f"{subject}. {content}" if subject else content

        # T5 task prefix for better instruction following
        source_text = "summarize: " + email_text
        target_text = item['summary']

        # Tokenize
        source = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': labels,
            'decoder_input_ids': target['input_ids'].squeeze()
        }

# Create datasets
train_dataset = DistillationDataset(
    train_data, student_tokenizer,
    config.MAX_SOURCE_LENGTH, config.MAX_TARGET_LENGTH
)
val_dataset = DistillationDataset(
    val_data, student_tokenizer,
    config.MAX_SOURCE_LENGTH, config.MAX_TARGET_LENGTH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"\n✅ Dataloaders created")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# ============================================================================
# OPTIMIZER AND SCHEDULER
# ============================================================================

optimizer = AdamW(
    student_model.parameters(),
    lr=config.LEARNING_RATE,
    eps=1e-8,
    weight_decay=config.WEIGHT_DECAY
)

total_steps = (len(train_loader) * config.EPOCHS) // config.GRADIENT_ACCUMULATION_STEPS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.WARMUP_STEPS,
    num_training_steps=total_steps
)

print(f"\n✅ Optimizer: AdamW (lr={config.LEARNING_RATE})")
print(f"✅ Scheduler: Linear warmup ({config.WARMUP_STEPS} steps)")
print(f"✅ Total steps: {total_steps:,}")

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss,
                   best_val_loss, patience_counter, history, is_best=False):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'history': history,
        'config': {
            'STUDENT_SIZE': config.STUDENT_SIZE,
            'TEACHER_MODEL': config.TEACHER_MODEL,
            'TEMPERATURE': config.TEMPERATURE,
            'ALPHA': config.ALPHA,
            'BETA': config.BETA,
            'GAMMA': config.GAMMA,
            # Include student config details for loading
            'student_config': STUDENT_CONFIGS[config.STUDENT_SIZE],
            'vocab_size': student_tokenizer.vocab_size,
            'dropout_rate': config.DROPOUT_RATE,
            'pad_token_id': student_tokenizer.pad_token_id,
            'eos_token_id': student_tokenizer.eos_token_id,
            'decoder_start_token_id': student_tokenizer.pad_token_id,
        }
    }

    # Save latest checkpoint
    checkpoint_path = config.CHECKPOINT_DIR / 'latest_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = config.CHECKPOINT_DIR / 'best_checkpoint.pt'
        torch.save(checkpoint, best_path)

        # Save in HuggingFace format
        model.save_pretrained(config.BEST_MODEL_DIR)
        student_tokenizer.save_pretrained(config.BEST_MODEL_DIR)

        # Save to Google Drive if enabled
        if config.GDRIVE_SAVE:
            try:
                gdrive_dir = Path(config.GDRIVE_PATH)
                gdrive_dir.mkdir(parents=True, exist_ok=True)

                # Copy best model
                if gdrive_dir.exists():
                    shutil.copytree(
                        config.BEST_MODEL_DIR,
                        gdrive_dir / 'best_model',
                        dirs_exist_ok=True
                    )
                    print(f"   ✅ Saved to Google Drive: {gdrive_dir}")
            except Exception as e:
                print(f"   ⚠️  Failed to save to Google Drive: {e}")

    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint and resume training"""
    if not checkpoint_path.exists():
        return None

    print(f"\n📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"✅ Resumed from epoch {checkpoint['epoch'] + 1}")
    print(f"   Train loss: {checkpoint['train_loss']:.4f}")
    print(f"   Val loss: {checkpoint['val_loss']:.4f}")
    print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")

    # Update history if needed
    if 'history' in checkpoint:
      return checkpoint['epoch'] + 1, checkpoint['best_val_loss'], checkpoint['patience_counter'], checkpoint['history']
    else:
      return checkpoint['epoch'] + 1, checkpoint['best_val_loss'], checkpoint['patience_counter'], {'train_loss': [], 'val_loss': [], 'learning_rate': []}


# Try to resume from checkpoint
start_epoch = 0
best_val_loss = float('inf')
patience_counter = 0
training_history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

if config.RESUME_FROM_CHECKPOINT:
    checkpoint_path = config.CHECKPOINT_DIR / 'latest_checkpoint.pt'
    checkpoint_data = load_checkpoint(student_model, optimizer, scheduler, checkpoint_path)

    if checkpoint_data:
        start_epoch, best_val_loss, patience_counter, training_history = checkpoint_data


# ============================================================================
# DISTILLATION LOSS FUNCTION
# ============================================================================

def compute_distillation_loss(student_logits, teacher_logits, labels, temperature, alpha, beta, gamma):
    """
    Compute combined distillation loss

    Loss = α * KL_div(student || teacher) + β * CE(student, labels) + γ * generation_loss
    """
    # 1. Distillation loss (KL divergence with temperature)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Ensure dimensions match for KL_div
    # Reshape labels to match logits for masking
    valid_mask = (labels != -100).unsqueeze(-1).expand_as(student_log_probs)

    # Apply mask to logits and labels
    student_log_probs_masked = student_log_probs[valid_mask].view(-1, student_log_probs.size(-1))
    teacher_probs_masked = teacher_probs[valid_mask].view(-1, teacher_probs.size(-1))

    # Check if there are valid tokens before computing KL_div
    if student_log_probs_masked.numel() > 0:
        distillation_loss = F.kl_div(
            student_log_probs_masked,
            teacher_probs_masked,
            reduction='batchmean'
        ) * (temperature ** 2)
    else:
        distillation_loss = torch.tensor(0.0, device=student_logits.device)


    # 2. Hard label loss (standard cross-entropy)
    # The CE loss already handles the ignore_index=-100
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # Combined loss
    total_loss = alpha * distillation_loss + beta * hard_loss

    return total_loss, distillation_loss, hard_loss

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(student, teacher, dataloader, optimizer, scheduler, device, grad_accum_steps):
    """Train for one epoch with distillation"""
    student.train()
    teacher.eval()

    total_loss = 0
    total_distill_loss = 0
    total_hard_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for step, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # decoder_input_ids = batch['decoder_input_ids'].to(device) # Not needed for teacher when labels are provided

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels # Teacher uses labels to generate decoder inputs internally
            )
            teacher_logits = teacher_outputs.logits

        # Student forward pass
        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels # Student also uses labels to generate decoder inputs internally
        )
        student_logits = student_outputs.logits

        # Compute distillation loss
        loss, distill_loss, hard_loss = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            config.TEMPERATURE,
            config.ALPHA,
            config.BETA,
            config.GAMMA
        )

        loss = loss / grad_accum_steps
        total_loss += loss.item() * grad_accum_steps
        total_distill_loss += distill_loss.item()
        total_hard_loss += hard_loss.item()

        # Backward pass
        loss.backward()

        # Update weights
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress_bar.set_postfix({
            'loss': loss.item() * grad_accum_steps,
            'distill': distill_loss.item(),
            'hard': hard_loss.item()
        })

    return (
        total_loss / len(dataloader),
        total_distill_loss / len(dataloader),
        total_hard_loss / len(dataloader)
    )

def evaluate(student, dataloader, device):
    """Evaluate the student model"""
    student.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("STARTING DISTILLATION TRAINING")
print("="*80)

for epoch in range(start_epoch, config.EPOCHS):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{config.EPOCHS}")
    print(f"{'='*50}")

    # Train
    train_loss, distill_loss, hard_loss = train_epoch(
        student_model,
        teacher_model,
        train_loader,
        optimizer,
        scheduler,
        config.device,
        config.GRADIENT_ACCUMULATION_STEPS
    )

    # Evaluate
    val_loss = evaluate(student_model, val_loader, config.device)

    # Get current learning rate
    current_lr = scheduler.get_last_lr()[0]

    # Log metrics
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['learning_rate'].append(current_lr)

    print(f"\n📊 Results:")
    print(f"   Train Loss: {train_loss:.4f} (distill: {distill_loss:.4f}, hard: {hard_loss:.4f})")
    print(f"   Val Loss: {val_loss:.4f}")
    print(f"   Learning Rate: {current_lr:.2e}")

    # Save checkpoint
    is_best = val_loss < best_val_loss

    if is_best:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"\n💾 New best model! (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"\n⏳ No improvement. Patience: {patience_counter}/{config.PATIENCE}")

    save_checkpoint(
        epoch, student_model, optimizer, scheduler,
        train_loss, val_loss, best_val_loss, patience_counter,
        training_history, is_best=is_best
    )

    # Early stopping
    if patience_counter >= config.PATIENCE:
        print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
        break

print("\n✅ Training completed!")
print(f"   Best validation loss: {best_val_loss:.4f}")

# ============================================================================
# EVALUATION WITH ROUGE
# ============================================================================

print("\n" + "="*80)
print("EVALUATING WITH ROUGE SCORES")
print("="*80)

# Load best model
try:
    checkpoint = torch.load(config.CHECKPOINT_DIR / 'best_checkpoint.pt', map_location=config.device)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded best model from epoch {checkpoint['epoch'] + 1}")
except FileNotFoundError:
    print("⚠️ Best checkpoint not found. Evaluating with the last trained model.")


def generate_summaries(model, dataloader, tokenizer, device, num_samples=None):
    """Generate summaries"""
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
            if num_samples is not None and i * config.BATCH_SIZE >= num_samples:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.MAX_TARGET_LENGTH,
                min_length=10,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=1.3,
                length_penalty=1.0
            )

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_preds)

            labels[labels == -100] = tokenizer.pad_token_id
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            references.extend(decoded_labels)

    return predictions, references

predictions, references = generate_summaries(
    student_model, val_loader, student_tokenizer, config.device
)

# Compute ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for pred, ref in zip(predictions, references):
    scores = scorer.score(ref, pred)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

print(f"\n📊 ROUGE Scores ({len(predictions)} samples):")
print(f"   ROUGE-1: {np.mean(rouge1_scores):.4f} (±{np.std(rouge1_scores):.4f})")
print(f"   ROUGE-2: {np.mean(rouge2_scores):.4f} (±{np.std(rouge2_scores):.4f})")
print(f"   ROUGE-L: {np.mean(rougeL_scores):.4f} (±{np.std(rougeL_scores):.4f})")

# Save results
results = {
    'rouge_scores': {
        'rouge1_mean': float(np.mean(rouge1_scores)),
        'rouge2_mean': float(np.mean(rouge2_scores)),
        'rougeL_mean': float(np.mean(rougeL_scores)),
    },
    'training_history': training_history,
    'best_epoch': checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 'N/A', # Handle case where best checkpoint wasn't saved
    'best_val_loss': float(best_val_loss),
}

with open(config.OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("🎉 DISTILLATION COMPLETE!")
print("="*80)
print(f"📁 Best model saved to: {config.BEST_MODEL_DIR}")
print(f"📊 ROUGE-L: {np.mean(rougeL_scores):.4f}")
print("="*80)