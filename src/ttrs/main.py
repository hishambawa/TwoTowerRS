from ttrs.modules import Model, DataLoader, RecommendationSystem, Towers

from ttrs.config.appsettings import AppSettings

from ttrs.utils.logger import BasicLogger

# initialize logger
logger = BasicLogger("debug")

logger.log_debug("Initialized the logger")

config = AppSettings()
data_loader = DataLoader(config)

TRAIN = False
rs = None

if TRAIN:
  # create the towers
    towers = Towers(data_loader=data_loader, embedding_dimension=config.embedding_dimension)

    # create the model
    model = Model(towers=towers, logger=logger, data_loader=data_loader, config=config)
    model.train_model()
    model.evaluate_model()

    rs = RecommendationSystem(model=model, data_loader=data_loader, logger=logger)
    rs.save()
  
else:
    rs = RecommendationSystem(data_loader=data_loader, logger=logger)

def print_recommendations(recommendations):
    print("== Recommendations ==\n")
    [print(f"{i+1}. {movie[0]} | [{ data_loader.get_genre(movie[1]) }] - ID: {movie[1]} - Score {movie[2]}") for i, movie in enumerate(recommendations)]
    print("\n=====================\n")

# run simulation
# set the initial movies the user likes and dislikes
liked_items = ["1"]
disliked_items = []

run = True
while run:
    # get the next 5 movies for the user
    recommendations = rs.get_recommendations(liked_items=liked_items, disliked_items=disliked_items)
    print_recommendations(recommendations)

    if len(recommendations) == 0:
        logger.log_info("No more recommendations available")
        break

    # active learning
    for recommendation in recommendations:
        user_input = input(f"Do you like {recommendation[0]} [{ data_loader.get_genre(recommendation[1]) }]? ")

        # quit if the input is q
        if user_input == "q":
            run = False
            break

        if user_input == "y":
            liked_items.append(recommendation[1])
        else:
            disliked_items.append(recommendation[1])