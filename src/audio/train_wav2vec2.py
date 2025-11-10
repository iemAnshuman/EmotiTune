import os
import yaml
import numpy as np
import torch
from datasets import Dataset, Audio, ClassLabel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.utils import setup_logger

# --- CONFIG & LOGGER ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger('wav2vec2_training', log_file='logs/wav2vec2_training.log')

# Define emotion mappings (same as before for consistency)
RAVDESS_EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
CREMA_EMOTIONS = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}

def get_file_paths_and_labels():
    """Scans directories to get all file paths and their labels."""
    paths = []
    labels = []

    # RAVDESS
    for subdir in ['ravdess_speech', 'ravdess_song']:
        full_path = os.path.join(PROJECT_ROOT, CONFIG['data']['raw'][subdir])
        if os.path.exists(full_path):
             for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith('.wav'):
                         parts = file.split('-')
                         if len(parts) >= 3:
                             emotion = RAVDESS_EMOTIONS.get(parts[2])
                             if emotion:
                                 paths.append(os.path.join(root, file))
                                 labels.append(emotion)

    # CREMA-D
    crema_path = os.path.join(PROJECT_ROOT, CONFIG['data']['raw']['crema_d'])
    if os.path.exists(crema_path):
        for file in os.listdir(crema_path):
            if file.endswith('.wav'):
                parts = file.split('_')
                if len(parts) >= 3:
                    emotion = CREMA_EMOTIONS.get(parts[2])
                    if emotion:
                        paths.append(os.path.join(crema_path, file))
                        labels.append(emotion)
    
    return paths, labels

def compute_metrics(eval_pred):
    """Computes accuracy and F1-score for Hugging Face Trainer."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(eval_pred.label_ids, predictions),
        'f1': f1_score(eval_pred.label_ids, predictions, average='weighted')
    }

def main():
    model_id = "facebook/wav2vec2-base"
    logger.info(f"Starting Wav2Vec2 training with base model: {model_id}")

    # 1. Prepare Data
    paths, labels = get_file_paths_and_labels()
    logger.info(f"Found {len(paths)} total audio files.")

    # Create ClassLabel to automatically handle string-to-int mapping
    label_feature = ClassLabel(names=sorted(list(set(labels))))
    
    # Split into train/val before creating Dataset objects
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=CONFIG['training']['val_split'], 
        random_state=CONFIG['training']['seed'], stratify=labels
    )

    # Create Hugging Face Datasets
    train_ds = Dataset.from_dict({'audio': train_paths, 'label': train_labels})
    val_ds = Dataset.from_dict({'audio': val_paths, 'label': val_labels})

    # Cast 'audio' column to Audio feature so it automatically loads and resamples
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=16000))

    # 2. Feature Extraction
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=16000 * 3, # Cap at 3 seconds to save memory
            truncation=True,
            padding=True 
        )
        # Map string labels to integers
        inputs["labels"] = [label_feature.str2int(label) for label in examples["label"]]
        return inputs

    logger.info("Preprocessing datasets...")
    encoded_train_ds = train_ds.map(preprocess_function, batched=True, batch_size=16, remove_columns=["audio", "label"])
    encoded_val_ds = val_ds.map(preprocess_function, batched=True, batch_size=16, remove_columns=["audio", "label"])

    # 3. Model Setup
    num_labels = len(label_feature.names)
    model = AutoModelForAudioClassification.from_pretrained(
        model_id, 
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_feature.names)},
        id2label={i: label for i, label in enumerate(label_feature.names)}
    )

    # 4. Training Arguments
    model_name = f"{model_id.split('/')[-1]}-finetuned-emotion"
    args = TrainingArguments(
        output_dir=f"models/{model_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5, # Lower learning rate for fine-tuning
        per_device_train_batch_size=8, # Smaller batch size due to larger model
        gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch
        per_device_eval_batch_size=8,
        num_train_epochs=CONFIG['training']['num_epochs'],
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_train_ds,
        eval_dataset=encoded_val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(PROJECT_ROOT, "models", "wav2vec2_emotion")
    trainer.save_model(final_path)
    logger.info(f"Model saved to {final_path}")

if __name__ == "__main__":
    main()