from dataclasses import dataclass

@dataclass
class AppSettings:
    # dataset config
    train_file_path: str = "data/processed/train_movielens_1m.tfrecord"
    test_file_path: str = "data/processed/test_movielens_1m.tfrecord"
    item_file_path: str = "data/movies.dat"
    train_batch_size: int = 12800
    test_batch_size: int = 2560
    max_context_length: int = 20

    # model config
    epochs: int = 3
    learning_rate: float = 0.1
    embedding_dimension: int = 32