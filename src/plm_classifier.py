import torch
from transformers import RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model
from logger import setup_logger, logger

logger = setup_logger(log_level="INFO")

def load_pretrained_model(pretrained_model_name='roberta-base', num_labels=2, use_peft=False):
    """
    Loads a pre-trained Roberta model for sequence classification, optionally with PEFT (LoRA).

    Args:
        pretrained_model_name (str): Name of the pre-trained model to load (e.g., 'roberta-base').
        num_labels (int): Number of output labels for classification.
        use_peft (bool): Whether to use PEFT (LoRA) for fine-tuning. Defaults to False (full fine-tuning).

    Returns:
        RobertaForSequenceClassification or PeftModel: The loaded Roberta model, potentially wrapped with PEFT.
    """
    try:
        logger.info(f"Loading pre-trained model: {pretrained_model_name}, PEFT: {use_peft}")
        model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

        if use_peft:
            logger.info("Setting up LoRA configuration...")
            peft_config = LoraConfig(
                r=8, # LoRA rank
                lora_alpha=32, # Scaling factor
                lora_dropout=0.05,
                target_modules=["query", "value"],
                bias="none",
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, peft_config)
            logger.info("PEFT (LoRA) config applied. Trainable parameters:")
            model.print_trainable_parameters()
        else:
            logger.info("Full fine-tuning will be used (PEFT is disabled).")

        logger.info(f"Pre-trained model '{pretrained_model_name}' loaded successfully with {num_labels} labels, PEFT: {use_peft}.")
        return model

    except Exception as e:
        logger.error(f"Error loading pre-trained model '{pretrained_model_name}' with PEFT={use_peft}: {e}")
        raise
