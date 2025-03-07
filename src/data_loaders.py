from torch.utils.data import DataLoader
from dataset import CDECDataset
from logger import setup_logger, logger

logger = setup_logger(log_level="INFO")

def create_data_loaders(train_data_path, dev_data_path, test_data_path, tokenizer_name, batch_size):
    """
    Creates DataLoaders for training, development, and testing datasets.
    Handles cases where some data paths might be dummy placeholders.
    """
    train_dataset, dev_dataset, test_dataset = None, None, None

    if train_data_path and train_data_path != 'dummy_train_path':
        logger.info(f"Creating Train Dataset from: {train_data_path}")
        try:
            train_dataset = CDECDataset(train_data_path, tokenizer_name, split='train', undersample_negative=True)
        except FileNotFoundError:
            logger.error(f"Train data file not found at: {train_data_path}. Please check the path.")
            raise
        logger.info("Train Dataset created.")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        logger.info("Train DataLoader created.")
    else:
        logger.info("Train DataLoader creation skipped.")
        train_dataloader = None

    if dev_data_path and dev_data_path != 'dummy_dev_path':
        logger.info(f"Creating Dev Dataset from: {dev_data_path}")
        try:
            dev_dataset = CDECDataset(dev_data_path, tokenizer_name, split='dev')
        except FileNotFoundError:
            logger.error(f"Dev data file not found at: {dev_data_path}. Please check the path.")
            raise
        logger.info("Dev Dataset created.")
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        logger.info("Dev DataLoader created.")
    else:
        logger.info("Dev DataLoader creation skipped.")
        dev_dataloader = None


    if test_data_path:
        logger.info(f"Creating Test Dataset from: {test_data_path}")
        try:
            test_dataset = CDECDataset(test_data_path, tokenizer_name, split='test')
        except FileNotFoundError:
            logger.error(f"Test data file not found at: {test_data_path}. Please check the path.")
            raise
        logger.info("Test Dataset created.")
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.info("Test DataLoader created.")
    else:
        logger.error("Test data path is missing. Evaluation cannot proceed without test data.")
        raise ValueError("Test data path is required for evaluation.")

    logger.info("All DataLoaders created.")
    return train_dataloader, dev_dataloader, test_dataloader
