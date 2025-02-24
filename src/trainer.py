import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, RobertaTokenizerFast
from plm_classifier import load_pretrained_model
from data_loaders import create_data_loaders
from logger import setup_logger, logger
import tqdm
import os

logger = setup_logger(log_level="DEBUG")

def train_model(train_data_path, dev_data_path, test_data_path,
                pretrained_model_name, tokenizer_name, batch_size,
                learning_rate, epochs, warmup_steps_ratio, weight_decay,
                device, use_peft=False, model_output_path='./trained_model'):
    """
    Fine-tunes a pre-trained Roberta model for event relation classification
    and saves the trained model
    """
    logger.info("--- Starting fine-tuning process ---")
    logger.info(f"  PEFT (LoRA) is {'enabled' if use_peft else 'disabled'}")
    logger.info(f"  Model saved to: {model_output_path}")

    logger.info("Loading DataLoaders...")
    train_dataloader, dev_dataloader, test_dataloader = create_data_loaders(
        train_data_path, dev_data_path, test_data_path, tokenizer_name, batch_size
    )
    logger.info("DataLoaders loaded successfully.")

    logger.info("Loading Pre-trained Model...")
    model = load_pretrained_model(pretrained_model_name, num_labels=2, use_peft=use_peft)
    model.to(device)
    logger.info(f"Pre-trained Model '{pretrained_model_name}' loaded and moved to {device}.")

    logger.info("Setting up Optimizer and Scheduler...")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_training_steps = len(train_dataloader) * epochs
    warmup_steps = int(warmup_steps_ratio * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    logger.info("Optimizer and Scheduler setup complete.")

    logger.info("--- Starting Training Loop ---")
    model.train()

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(train_dataloader, total=len(train_dataloader),
                                 desc=f"Epoch {epoch + 1} Training")

        for batch in progress_bar:
            input_ids_sentence1 = batch['input_ids_sentence1'].squeeze(1).to(device)
            attention_mask_sentence1 = batch['attention_mask_sentence1'].squeeze(1).to(device)
            input_ids_sentence2 = batch['input_ids_sentence2'].squeeze(1).to(device)
            attention_mask_sentence2 = batch['attention_mask_sentence2'].squeeze(1).to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=torch.cat((input_ids_sentence1, input_ids_sentence2), dim=1),
                attention_mask=torch.cat((attention_mask_sentence1, attention_mask_sentence2), dim=1),
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss:.4f}")

    logger.info("--- Training Loop Completed ---")

    # 6. Save Trained Model
    logger.info(f"Saving trained model to: {model_output_path}")
    os.makedirs(model_output_path, exist_ok=True)
    model.save_pretrained(model_output_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(model_output_path)
    logger.info(f"Trained model and tokenizer saved to {model_output_path} successfully.")


    return model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler


if __name__ == '__main__':
    train_data_path = 'llm-CDEC/data/event_pairs.train'
    dev_data_path = 'llm-CDEC/data/event_pairs.dev'
    test_data_path = 'llm-CDEC/data/event_pairs.test'
    pretrained_model_name = 'roberta-base'
    tokenizer_name = 'roberta-base'
    batch_size = 16
    learning_rate = 2e-5
    epochs = 1
    warmup_steps_ratio = 0.1
    weight_decay = 0.01
    use_peft = True
    model_output_path = './trained_event_relation_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        model, train_loader, dev_loader, test_loader, optimizer, scheduler = train_model(
            train_data_path, dev_data_path, test_data_path,
            pretrained_model_name, tokenizer_name, batch_size,
            learning_rate, epochs, warmup_steps_ratio, weight_decay,
            device, use_peft=use_peft, model_output_path=model_output_path
        )
        logger.info("Fine-tuning process completed successfully!")
        logger.info(f"Trained model saved to: {model_output_path}")

    except Exception as e:
        logger.critical(f"Fine-tuning process setup failed: {e}")