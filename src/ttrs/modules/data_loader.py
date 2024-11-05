from seqrec.config.appsettings import AppSettings

import tensorflow as tf
import numpy as np
import pandas as pd

class DataLoader:

    def __init__(self, config: AppSettings):
        self.config = config
        self.trainset, self.testset = self._process_train_test_data()
        self.items, self.item_ids, self.item_data, self.raw_item_data = self._process_movie_data()

    def get_items(self):
        return self.items

    def get_unique_movie_ids(self):
        return self.item_ids
    
    def get_title(self, item_id):
        return self.item_data[item_id][0]
    
    def get_genre(self, item_id):
        return self.item_data[item_id][1]
    
    def _process_train_test_data(self):
        train = tf.data.TFRecordDataset(self.config.train_file_path)
        test = tf.data.TFRecordDataset(self.config.test_file_path)
    
        train_ds = train.map(self._parse_data).map(lambda x: {
            "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
            "label_movie_id": tf.strings.as_string(x["label_movie_id"]),
            "context_movie_genre": tf.strings.as_string(x["context_movie_genre"])
        })

        test_ds = test.map(self._parse_data).map(lambda x: {
            "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
            "label_movie_id": tf.strings.as_string(x["label_movie_id"]),
            "context_movie_genre": tf.strings.as_string(x["context_movie_genre"])
        })

        # create train test cache
        cached_train = train_ds.shuffle(10_000).batch(self.config.train_batch_size).cache()
        cached_test = test_ds.batch(self.config.test_batch_size).cache()

        # clean up the variables
        del train, test, train_ds, test_ds

        return cached_train, cached_test
    
    def _parse_data(self, example_proto):
        feature_description = {
            'context_movie_id': tf.io.FixedLenFeature([self.config.max_context_length], tf.int64, default_value=np.repeat(0, self.config.max_context_length)),
            'context_movie_rating': tf.io.FixedLenFeature([self.config.max_context_length], tf.float32, default_value=np.repeat(0, self.config.max_context_length)),
            'context_movie_year': tf.io.FixedLenFeature([self.config.max_context_length], tf.int64, default_value=np.repeat(1980, self.config.max_context_length)),
            'context_movie_genre': tf.io.FixedLenFeature([self.config.max_context_length], tf.string, default_value=np.repeat("Drama", self.config.max_context_length)),
            'label_movie_id': tf.io.FixedLenFeature([1], tf.int64, default_value=0)
        }
        return tf.io.parse_single_example(example_proto, feature_description)
    
    def _process_movie_data(self): 
        raw_movies_df = pd.read_csv(self.config.item_file_path,
                            sep="::",
                            header=None,
                            names=["movie_id", "movie_title", "genre"],
                            dtype={"movie_id": str, "movie_title": str, "genre": str},
                            engine="python")
        
        raw_movies = tf.data.Dataset.from_tensor_slices(dict(raw_movies_df))
        movies = raw_movies.map(lambda x: x["movie_id"])
        movie_ids = movies.batch(1_000)
        unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))

        # preload the movie id to title mappings
        # Use .iterrows() to iterate over rows and access data using column names
        data = {row["movie_id"]: (row["movie_title"], row["genre"])
                            for _, row in raw_movies_df.iterrows()}
        
        return movies, unique_movie_ids, data, raw_movies_df