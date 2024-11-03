import tensorflow_recommenders as tfrs
import tensorflow as tf

from config.appsettings import AppSettings
from modules.data_loader import DataLoader
from modules.towers import Towers
from utils.logger import BasicLogger

class Model(tfrs.Model):

    def __init__(self, towers: Towers, data_loader: DataLoader, logger: BasicLogger, config: AppSettings):
        super().__init__()

        self.data_loader = data_loader
        self.logger = logger
        self.config = config

        self._query_model = towers.get_query_tower()
        self._candidate_model = towers.get_candidate_tower()
        self._task = towers.get_loss_function(candidate_model=self._candidate_model)

    def compute_loss(self, features, training=False):
        watch_history = features["context_movie_id"]
        watch_next_label = features["label_movie_id"]

        query_embedding = self._query_model(watch_history)
        candidate_embedding = self._candidate_model(watch_next_label)

        return self._task(query_embedding, candidate_embedding, compute_metrics=not training)
    
    def train_model(self):
        self.logger.log_debug("Training the model")

        self.compile(optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate=self.config.learning_rate))
        self.fit(self.data_loader.trainset, epochs=self.config.epochs)

        self.logger.log_debug("Model trained successfully")

    def evaluate_model(self):
        self.logger.log_debug("Evaluating the model")

        metrics = self.evaluate(self.data_loader.testset)

        self.logger.log_debug(f"Metrics: {metrics}")
        self.logger.log_debug("Model evaluated successfully")