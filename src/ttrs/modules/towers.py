import tensorflow as tf
import tensorflow_recommenders as tfrs

from seqrec.modules.data_loader import DataLoader

class Towers:

    def __init__(self, data_loader: DataLoader, embedding_dimension):
        self.unique_item_ids = data_loader.get_unique_movie_ids()
        self.items = data_loader.get_items()
        self.embedding_dimension = embedding_dimension

    def get_query_tower(self):
        return tf.keras.Sequential([
            tf.keras.layers.StringLookup(
            vocabulary=self.unique_item_ids, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_item_ids) + 1, self.embedding_dimension),
            tf.keras.layers.GRU(self.embedding_dimension),
        ])
    
    def get_candidate_tower(self):
        return tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_item_ids, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_item_ids) + 1, self.embedding_dimension)
        ])

    def get_loss_function(self, candidate_model):
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=self.items.batch(128).map(candidate_model),
            ks=[1, 5, 10, 20, 50, 100]
        )

        task = tfrs.tasks.Retrieval(
            metrics=metrics
        )

        return task
