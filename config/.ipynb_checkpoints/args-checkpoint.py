from dataclasses import dataclass


@dataclass
class Args:
    """
    Training arguments.
    """

    # Split ratio for train and val set
    split_ratio: float = 0.9
    # Trainign batch size
    batch_size: int = 4
    # Maximum chunk length
    block_size: int = 8
