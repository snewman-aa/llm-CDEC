from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from logger import setup_logger, logger
from tqdm import tqdm

logger = setup_logger(log_level="INFO")

class CDECDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, split='train'):
        """
        Args:
            data_path (str): Path to the data file.
            tokenizer_name (str): Name of the tokenizer to use (e.g., 'roberta-base').
            split (str):  Dataset split ('train', 'dev', 'test'). Defaults to 'train'.
        """
        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.split = split
        self.data = self.load_and_preprocess_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_and_preprocess_data(self):
        logger.info(f"Loading and preprocessing data from: {self.data_path} for {self.split} split...")
        processed_data = []
        line_count = 0
        total_lines = 0

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            logger.info(f"Processing {total_lines} lines from {self.data_path}...")

            for line in tqdm(lines, desc=f"Processing {self.split} data", unit="line"):
                line_count += 1
                sample = self.process_line(line.strip().split('\t'), line_count)
                processed_data.append(sample)

        logger.info(f"Finished processing {self.split} data.")
        logger.info(f"Loaded {len(processed_data)} samples for {self.split} split.")
        return processed_data

    def process_line(self, fields, line_count):
        """Processes a single line from the data file, handling different splits."""
        id_fields_present = (self.split == 'test')

        try:
            if id_fields_present:
                event1_id = fields[0]
                event2_id = fields[1]
                sentence1_index_offset = 2
                sentence2_index_offset = 13
                label_index = 24
            else:
                event1_id = f"{self.split[:3]}_{line_count:03d}_11#{line_count:03d}_1_1"
                event2_id = f"{self.split[:3]}_{line_count:03d}_22#{line_count:03d}_2_2"
                sentence1_index_offset = 0
                sentence2_index_offset = 11
                label_index = 22


            sentence1 = fields[sentence1_index_offset]
            event1_trigger_start = int(fields[sentence1_index_offset + 1])
            event1_trigger_end = int(fields[sentence1_index_offset + 2])

            sentence2 = fields[sentence2_index_offset]
            event2_trigger_start = int(fields[sentence2_index_offset + 1])
            event2_trigger_end = int(fields[sentence2_index_offset + 2])

            label = int(fields[label_index])

            tokenized_sentence1 = self.tokenizer(sentence1, truncation=True, padding='max_length', max_length=128)
            tokenized_sentence2 = self.tokenizer(sentence2, truncation=True, padding='max_length', max_length=128)

            sample = {
                'event1_id': event1_id,
                'event2_id': event2_id,
                'sentence1': sentence1,
                'sentence2': sentence2,
                'input_ids_sentence1': tokenized_sentence1['input_ids'],
                'attention_mask_sentence1': tokenized_sentence1['attention_mask'],
                'input_ids_sentence2': tokenized_sentence2['input_ids'],
                'attention_mask_sentence2': tokenized_sentence2['attention_mask'],
                'label': label
            }
            return sample

        except ValueError as e:
            logger.error(f"({self.split.capitalize()} Split) Error processing line: {fields} - {e}")
            raise e


if __name__ == '__main__':
    logger = setup_logger(log_level="DEBUG")

    train_set_path = '../data/event_pairs.train'
    dev_set_path = '../data/event_pairs.dev'
    test_set_path = '../data/event_pairs.test'

    train_dataset = CDECDataset(data_path=train_set_path, tokenizer_name='roberta-base', split='train')
    dev_dataset = CDECDataset(data_path=dev_set_path, tokenizer_name='roberta-base', split='dev')
    test_dataset = CDECDataset(data_path=test_set_path, tokenizer_name='roberta-base', split='test')

    logger.debug("\n--- Train Sample ---")
    sample_train = train_dataset[0]
    logger.debug("Sample Data (Train): {}", sample_train)
    logger.info("Number of samples (Train): {}", len(train_dataset))

    logger.debug("\n--- Dev Sample ---")
    sample_dev = dev_dataset[0]
    logger.debug("Sample Data (Dev): {}", sample_dev)
    logger.info("Number of samples (Dev): {}", len(dev_dataset))

    logger.debug("\n--- Test Sample ---")
    sample_test = test_dataset[0]
    logger.debug("Sample Data (Test): {}", sample_test)
    logger.info("Number of samples (Test): {}", len(test_dataset))