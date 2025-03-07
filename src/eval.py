import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from peft import PeftModel, PeftConfig
from data_loaders import create_data_loaders
from logger import setup_logger, logger
from sklearn.metrics import classification_report
import tqdm

logger = setup_logger(log_level="DEBUG")

def evaluate_model(test_data_path,
                   tokenizer_name,
                   model_repo_or_path,
                   batch_size, device, use_peft=False):
    """
    Evaluates a trained Roberta model for event relation classification on the test set.

    Args:
        test_data_path (str): Path to the test data file
        tokenizer_name (str): Name of the tokenizer to use (e.g., 'roberta-base')
        model_repo_or_path (str):  Hugging Face Hub repo ID
        batch_size (int): Batch size for evaluation
        device (torch.device): 'cuda' or 'cpu' device to use
        use_peft (bool): Whether PEFT/LoRA was used for fine-tuning. Defaults to False

    Returns:
        dict: Dictionary containing evaluation metrics (e.g., F1-score, Precision, Recall, Accuracy).
    """
    logger.info("--- Starting Model Evaluation ---")
    logger.info(f"  Model Repo/Path: {model_repo_or_path}, PEFT: {use_peft}")

    logger.info("Loading Test DataLoader...")
    _, _, test_dataloader = create_data_loaders(
        'dummy_train_path', 'dummy_dev_path', test_data_path, tokenizer_name, batch_size
    )
    logger.info("Test DataLoader loaded successfully.")

    logger.info("Loading Trained Model and Tokenizer...")
    if use_peft:
        peft_config = PeftConfig.from_pretrained(model_repo_or_path)
        model = RobertaForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=2)
        model = PeftModel.from_pretrained(model, model_repo_or_path)
    else:
        model = RobertaForSequenceClassification.from_pretrained(model_repo_or_path, num_labels=2)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_repo_or_path)
    model.to(device)
    logger.info(f"Trained Model and Tokenizer loaded from '{model_repo_or_path}' and moved to {device}.")

    logger.info("--- Starting Evaluation Loop ---")
    model.eval()

    true_labels = []
    predicted_labels = []

    progress_bar = tqdm.tqdm(test_dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids_sentence1 = batch['input_ids_sentence1'].squeeze(1).to(device)
            attention_mask_sentence1 = batch['attention_mask_sentence1'].squeeze(1).to(device)
            input_ids_sentence2 = batch['input_ids_sentence2'].squeeze(1).to(device)
            attention_mask_sentence2 = batch['attention_mask_sentence2'].squeeze(1).to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=torch.cat((input_ids_sentence1, input_ids_sentence2), dim=1),
                attention_mask=torch.cat((attention_mask_sentence1, attention_mask_sentence2), dim=1)
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    logger.info("Calculating Evaluation Metrics...")
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    positive_f1 = report['1']['f1-score']
    accuracy = report['accuracy']
    logger.info("Evaluation Metrics calculated.")

    metrics = {
        'positive_f1': positive_f1,
        'accuracy': accuracy,
        'classification_report': report
    }

    logger.info("--- Evaluation Completed ---")
    logger.info(f"  F1 Score (Positive Class): {positive_f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.debug(f"  Full Classification Report:\n{report}")

    return metrics


if __name__ == '__main__':
    test_data_path = 'llm-CDEC/data/event_pairs.test'
    tokenizer_name = 'roberta-base'
    model_repo_or_path = "snewman/roberta-base-CDEC"
    batch_size = 32
    use_peft = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for evaluation: {device}")

    try:
        evaluation_metrics = evaluate_model(
            test_data_path, tokenizer_name, model_repo_or_path, batch_size, device, use_peft=use_peft
        )
        logger.info("Model evaluation completed successfully!")

    except Exception as e:
        logger.critical(f"Model evaluation failed: {e}")
