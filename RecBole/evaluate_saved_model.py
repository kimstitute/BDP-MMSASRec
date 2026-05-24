import sys
import torch
from recbole.utils import get_trainer, init_seed, init_logger, get_model
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from logging import getLogger

def evaluate_checkpoint(model_file):
    # Load checkpoint
    try:
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_file, map_location='cpu')
    config = checkpoint["config"]

    # Override for stability
    config['worker'] = 0  # Use main process to avoid shm issues in test
    config['eval_batch_size'] = 4096
    config['state'] = 'INFO'

    # Initialize
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    
    logger.info("Restoring from checkpoint: {}".format(model_file))
    logger.info("Overriding worker to 0 and eval_batch_size to 4096 for stability.")

    # Dataset
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    # Trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Test
    logger.info("Starting Test Evaluation...")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    
    logger.info("Test Result: {}".format(test_result))
    return test_result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_saved_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    evaluate_checkpoint(model_path)
