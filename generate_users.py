import pandas as pd
import numpy as np
import random
import joblib

# кол-во пользователей
N_USERS = 1000
# диапазон кол-ва треков на одного пользователя
MIN_TRACKS = 10
MAX_TRACKS = 30
# доля жанровых пользователей
GENRE_USERS_RATIO = 0.7
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# загрузка данных
X_test = pd.read_pickle("X_test.pkl")
genre_cols = joblib.load("genre_cols.pkl")
genres_test = X_test[genre_cols]

# схожие жанры
genre_neighbors = {
    "Rock": ["Pop", "Electronic", "Experimental", "Hip-Hop"],
    "Pop": ["Rock", "Electronic", "Hip-Hop", "Soul-RnB"],
    "Jazz": ["Blues", "Soul-RnB", "Instrumental", "Classical"],
    "Blues": ["Jazz", "Soul-RnB", "Country"],
    "Electronic": ["Rock", "Experimental", "Instrumental", "Pop"],
    "Hip-Hop": ["Pop", "Soul-RnB", "Electronic"],
    "Classical": ["Instrumental", "Easy Listening"],
    "Folk": ["Country", "International", "Old-Time / Historic"],
    "Country": ["Folk", "Blues"],
    "Old-Time / Historic": ["Folk", "Country", "Blues"],
    "Instrumental": ["Classical", "Electronic", "Experimental"],
    "Experimental": ["Electronic", "Rock"],
    "Soul-RnB": ["Jazz", "Hip-Hop", "Pop"],
    "International": ["Folk", "Spoken"],
    "Spoken": ["International"],
    "Easy Listening": ["Classical", "Instrumental"]
}

all_tracks = genres_test.index.tolist()
# жанры без Other
valid_genres = [g for g in genre_neighbors.keys()]

# выбор треков, которые относятся хотя бы к одному
# из заданных жанров
def get_tracks_by_genres(selected_genres):
    mask = genres_test[selected_genres].sum(axis=1) > 0
    return genres_test[mask].index.tolist()

# набор из 3-5 жанров для жанрового пользователя
def build_genre_user():

    seed = random.choice(valid_genres)
    selected = set([seed])

    n_genres = random.randint(3, 5)

    while len(selected) < n_genres:

        current = random.choice(list(selected))
        neighbors = genre_neighbors.get(current, [])

        if not neighbors:
            continue

        selected.add(random.choice(neighbors))

    return list(selected)[:n_genres]


users = []
# кол-во жанровых пользователей
n_genre_users = int(N_USERS * GENRE_USERS_RATIO)
# кол-во случайных пользователей
n_random_users = N_USERS - n_genre_users

# айди жанрового пользователя
genre_user_id = 0
# создание жанровых пользователей
while len([u for u in users if u["type"] == "genre"]) < n_genre_users:

    selected_genres = build_genre_user()

    candidate_tracks = get_tracks_by_genres(selected_genres)

    if len(candidate_tracks) < MIN_TRACKS:
        continue

    n_tracks = random.randint(MIN_TRACKS, MAX_TRACKS)
    n_tracks = min(n_tracks, len(candidate_tracks))

    liked_tracks = random.sample(candidate_tracks, n_tracks)

    users.append({
        "user_id": f"genre_user_{genre_user_id}",
        "type": "genre",
        "genres": selected_genres,
        "liked_tracks": liked_tracks
    })

    genre_user_id += 1

# создание случайных пользователей
for i in range(n_random_users):

    n_tracks = random.randint(MIN_TRACKS, MAX_TRACKS)

    liked_tracks = random.sample(all_tracks, n_tracks)

    users.append({
        "user_id": f"random_user_{i}",
        "type": "random",
        "genres": None,
        "liked_tracks": liked_tracks
    })

#сохранение
joblib.dump(users, "users.pkl")
print("Generated users:", len(users))
print("Saved to users.pkl")