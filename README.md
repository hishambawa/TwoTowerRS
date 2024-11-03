# TwoTowerRS 
## A sequential Recommendation System built using the Two Tower Embedding model

Run the data generator
```
python3 -m datagen_min --data_dir=data  --output_dir=data/processed  --min_timeline_length=3  --max_context_length=10  --max_context_movie_genre_length=10  --min_rating=2
```