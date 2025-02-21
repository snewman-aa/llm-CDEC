import torch
from transformers import RobertaForSequenceClassification
from logger import setup_logger, logger

logger = setup_logger(log_level="INFO")

def load_pretrained_model(pretrained_model_name='roberta-base', num_labels=2):
    """
    Loads a pre-trained Roberta model for sequence classification.

    Args:
        pretrained_model_name (str): Name of the pre-trained model to load (e.g., 'roberta-base').
        num_labels (int): Number of output labels for classification.

    Returns:
        RobertaForSequenceClassification: The loaded Roberta model.
    """
    try:
        logger.info(f"Loading pre-trained model: {pretrained_model_name}")
        model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)
        logger.info(f"Pre-trained model '{pretrained_model_name}' loaded successfully with {num_labels} labels.")
        return model
    except Exception as e:
        logger.error(f"Error loading pre-trained model '{pretrained_model_name}': {e}")
        raise e
