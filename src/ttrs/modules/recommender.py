import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

from seqrec.utils.logger import BasicLogger
from seqrec.modules.data_loader import DataLoader

class RecommendationSystem:
    def __init__(self, data_loader: DataLoader, logger:BasicLogger, model=None):
        self.data_loader = data_loader
        self.logger = logger
        
        # if a model is passed, use it to create the index
        # else load the model from disk
        if model is None:
            self.logger.log_info("Model not provided. Loading the model from disk")
            self.load()
        else:
            self.logger.log_info("Model provided. Creating the index")
            # create a model that takes in raw query features
            self.index = tfrs.layers.factorized_top_k.BruteForce(model._query_model)

            # recommends movies out of the entire movies dataset.
            movies = self.data_loader.get_items()

            self.index.index_from_dataset(
                tf.data.Dataset.zip(
                    (movies.batch(100), movies.batch(100).map(model._candidate_model))
                )
            )

            del movies

            self.logger.log_info("Index created successfully")

    def get_recommendations(self, liked_items, disliked_items, context_length = 5, k = 5):
        # reshape the items to match the expected input shape for the index
        processed_list = np.array(liked_items[-context_length:]).reshape(-1, 1)

        # convert the list of rated items into a TensorFlow constant
        query = tf.constant(processed_list)

        # use the index to get recommended movie titles based on the query
        scores, titles = self.index(query)

        # append the liked and disliked items to not recommend them again
        rated_items = liked_items + disliked_items

        # convert the list of watched movies into a set for efficient lookup
        rated_set = set(rated_items)

        # convert the recommended movie titles from tensor to a list
        recommended_movies = titles.numpy()[0].tolist()
        score = scores.numpy()[0].tolist()

        # decode movie titles from bytes to strings and filter out those already watched
        # convert the movie id to title and get the top 5
        result = []
        i = 0

        for raw_movie_id in recommended_movies:
            # decode the movie ID from bytes to string
            movie_id = raw_movie_id.decode('UTF-8')

            # check if the movie has already been watched
            if movie_id not in rated_set:
                # get the movie title and append the tuple (movie_id, movie_title) to the result
                result.append((self.data_loader.get_title(movie_id), movie_id, score[i]))
                i += 1

            # stop after collecting K movies
            if len(result) == k:
                break

        
        # extract movie_ids from the results list
        movie_ids = [movie_id for _, movie_id, _ in result]

        # convert the final results to a dataframe
        # based on the item data
        result_df = self.data_loader.raw_item_data[self.data_loader.raw_item_data['movie_id'].isin(movie_ids)]

        return result_df[['movie_title', 'movie_id', 'genre']]

    def save(self):
      # the model must be called atleast once before saving
      # else if will throw an error when loading and inferring
      # this is a bug in tensorflow
      self.index(tf.constant([["1"]]))

      tf.saved_model.save(self.index, "data/predictions")
      self.logger.log_info("Successfully saved the model")

    def load(self):
      self.index = tf.saved_model.load("data/predictions")
      self.logger.log_info("Successfully loaded the model")
