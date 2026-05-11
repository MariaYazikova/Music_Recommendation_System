import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# путь к аудио признакам
FEATURES_PATH = "fma_metadata/features.csv"
# путь к метаданным
TRACKS_PATH = "fma_metadata/tracks.csv"
# путь к жанрам
GENRES_PATH = "fma_metadata/genres.csv"

# загрузка аудио признаков
features = pd.read_csv(
    FEATURES_PATH,
    index_col=0,
    header=[0, 1, 2]
)
print("Original shape:", features.shape)

# загрузка метаданных
tracks = pd.read_csv(
    TRACKS_PATH,
    index_col=0,
    header=[0, 1]
)

# загрузка таблицы жанров
genres_df = pd.read_csv(GENRES_PATH)

# получение top_level (parent) жанра к треку
genre_to_top = genres_df.set_index("genre_id")["top_level"]
genre_to_title = genres_df.set_index("genre_id")["title"]
genre_ids = tracks[("track", "genres")]
genre_ids = genre_ids.apply(
    lambda x: ast.literal_eval(x)
    if pd.notna(x)
    else []
)

def get_top_level_genres(ids):

    top_genres = []

    for genre_id in ids:

        if genre_id not in genre_to_top.index:
            continue

        top_id = genre_to_top[genre_id]

        if top_id in genre_to_title.index:

            top_genres.append(
                genre_to_title[top_id]
            )

    # если ничего не нашли присваиваем Other
    if len(top_genres) == 0:
        return ["Other"]

    return list(set(top_genres))

top_genres = genre_ids.apply(get_top_level_genres)

# сколько треков попало в Other
other_count = top_genres.apply(
    lambda x: x == ["Other"]
).sum()

print("Tracks with Other genre:", other_count)

# multi-hot encoding parent жанров
genres_exploded = top_genres.explode()
genres_encoded = (
    pd.get_dummies(genres_exploded)
    .groupby(level=0)
    .max()
    .astype(int)
)

print("Genres encoded shape:", genres_encoded.shape)

# преобразование multiindex колонок в плоские строки
def flatten_columns(cols):
    return [
        "_".join(map(str, c)) if isinstance(c, tuple) else str(c)
        for c in cols
    ]
features.columns = flatten_columns(features.columns)

# разбиение датасета с аудио на train и temp
X_train_audio, X_temp_audio = train_test_split(
    features,
    test_size=0.3,
    random_state=42
)
#разбиение temp на val и test
X_val_audio, X_test_audio = train_test_split(
    X_temp_audio,
    test_size=0.5,
    random_state=42
)

# жанры теми же индексами
genres_train = genres_encoded.loc[X_train_audio.index]
genres_val = genres_encoded.loc[X_val_audio.index]
genres_test = genres_encoded.loc[X_test_audio.index]

# масштабирование
scaler = StandardScaler()

X_train_audio_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_audio),
    index=X_train_audio.index,
    columns=X_train_audio.columns
)

X_val_audio_scaled = pd.DataFrame(
    scaler.transform(X_val_audio),
    index=X_val_audio.index,
    columns=X_val_audio.columns
)

X_test_audio_scaled = pd.DataFrame(
    scaler.transform(X_test_audio),
    index=X_test_audio.index,
    columns=X_test_audio.columns
)

# объединение с жанрами
X_train = pd.concat(
    [X_train_audio_scaled, genres_train],
    axis=1
)

X_val = pd.concat(
    [X_val_audio_scaled, genres_val],
    axis=1
)

X_test = pd.concat(
    [X_test_audio_scaled, genres_test],
    axis=1
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

# сохранение scaler
joblib.dump(scaler, "scaler.pkl")

# сохранение данных
X_train.to_pickle("X_train.pkl")
X_val.to_pickle("X_val.pkl")
X_test.to_pickle("X_test.pkl")
joblib.dump(features.columns, "audio_cols.pkl")
joblib.dump(genres_encoded.columns, "genre_cols.pkl")

print("\nDone: data prepared")