from dataclasses import dataclass


@dataclass
class Args:
    """
    Training hyperparameters.
    """

    # Split ratio for train and val set
    split_ratio: float = 0.9
    # Trainign batch size
    batch_size: int = 32
    # Maximum chunk length
    block_size: int = 8
    # Total training epochs
    max_iters: int = 3000
    # Evaluate after
    eval_interval: int = 300
    # Learning rate
    learning_rate: float = 1e-2
