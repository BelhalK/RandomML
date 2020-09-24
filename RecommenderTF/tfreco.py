import tensorflow as tf
 
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from MovieLens import TwoTowerMovielensModel


print("==> Loading Ratings...")
# Ratings data.
ratings = tfds.load("movie_lens/100k-ratings", split="train")

print("==> Loading Movies...")
# Features of all the available movies.
movies = tfds.load("movie_lens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])


print("==> Creating Recommender Model...")
model = TwoTowerMovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
 

print("==> Training Phase...")
model.fit(ratings.batch(4096), verbose=False)


index = tfrs.layers.ann.BruteForce(model.user_model)
index.index(movies.batch(100).map(model.movie_model), movies)
 

# Get recommendations.
user_id = "19"
print(f"==> Inference Phase... for User {user_id}")
_, titles = index(tf.constant([user_id]))
print(f"Recommendations for User {user_id}: {titles[0, :3]}")