import torch
from torch.utils.data import DataLoader
from dataset import CDECDataset
from logger import setup_logger, logger

logger = setup_logger(log_level="INFO")

def create_data_loaders(train_data_path, dev_data_path, test_data_path, tokenizer_name, batch_size=32):
    """
    Creates DataLoader instances for training, development, and test sets.

    Args:
        train_data_path (str): Path to the training data file.
        dev_data_path (str): Path to the development data file.
        test_data_path (str): Path to the test data file.
        tokenizer_name (str): Name of the tokenizer to use.
        batch_size (int): Batch size for DataLoader. Defaults to 32.

    Returns:
        tuple: (train_dataloader, dev_dataloader, test_dataloader)
               Tuple containing DataLoader objects for train, dev, and test sets.
    """
    logger.info("Creating Datasets...")
    train_dataset = CDECDataset(data_path=train_data_path, tokenizer_name=tokenizer_name, split='train')
    dev_dataset = CDECDataset(data_path=dev_data_path, tokenizer_name=tokenizer_name, split='dev')
    test_dataset = CDECDataset(data_path=test_data_path, tokenizer_name=tokenizer_name, split='test')
    logger.info("Datasets created successfully.")

    logger.info(f"Creating DataLoaders with batch size: {batch_size}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("DataLoaders created successfully.")

    return train_dataloader, dev_dataloader, test_dataloader
