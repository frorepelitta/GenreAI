import os, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# === БЛОК 1. ПОДГОТОВКА ДАННЫХ ===
def load_data():
    """Загружает CSV (movie_genre_classification_final.csv и др.) или IMDb train_data.txt."""
    paths = [
        "data/movie_genre_classification_final.csv", "movie_genre_classification_final.csv",
        "data/train_data.txt", "train_data.txt",
        "data/movies.csv", "movies.csv", "data/books.csv", "books.csv",
    ]
    for path in paths:
        if os.path.exists(path):
            if path.endswith(".txt"):
                rows = [line.split(":::") for line in open(path, encoding="utf-8", errors="ignore") if line.count(":::") >= 3]
                df = pd.DataFrame({"genre": [r[2].strip() for r in rows], "text": [r[3].strip() for r in rows]})
            else:
                df = pd.read_csv(path)
                cols = {c.lower(): c for c in df.columns}
                g_col = next(cols[k] for k in ["genre", "genres", "category"] if k in cols)
                t_col = next(cols[k] for k in ["short plot summary", "description", "plot", "overview", "summary", "text"] if k in cols)
                df = df[[g_col, t_col]].rename(columns={g_col: "genre", t_col: "text"})
            df = df.dropna(subset=["genre", "text"])
            df["genre"] = df["genre"].astype(str).str.strip().str.lower()
            df["text"] = df["text"].astype(str).str.strip()
            df = df[df["text"].str.len() > 20]
            top_genres = df["genre"].value_counts().head(7).index
            df = df[df["genre"].isin(top_genres)].reset_index(drop=True)
            if len(df) > 15000:  # Ограничиваем объём для быстрого обучения на обычном ПК
                df, _ = train_test_split(df, train_size=15000, stratify=df["genre"], random_state=42)
            return df.reset_index(drop=True)
    raise FileNotFoundError("Положите датасет (movie_genre_classification_final.csv или train_data.txt) в папку data/.")

df = load_data()

# Признак (X) — текст описания сюжета (TF-IDF с norm="l2" переводит текст в числовую матрицу
# и масштабирует вектор каждого описания к единой длине, чтобы длинные тексты не перевешивали короткие).
# Целевая переменная (y) — жанр фильма/книги, кодируем словарём GENRE_TO_ID в числа 0..N-1.
genres = sorted(df["genre"].unique())
GENRE_TO_ID = {g: i for i, g in enumerate(genres)}
ID_TO_GENRE = {i: g for g, i in GENRE_TO_ID.items()}
X, y = df["text"].values, df["genre"].map(GENRE_TO_ID).values

# Делим данные на 80% (обучение и кросс-валидация) и 20% (отложенный тест "в сейф") со stratify=y,
# чтобы пропорции всех жанров в обучении и итоговой проверке совпадали.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === БЛОК 2. ОБУЧЕНИЕ И ДИАГНОСТИКА (2 принципиально разных алгоритма + CV) ===
models = {
    "LogisticRegression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", norm="l2")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
    "MultinomialNB": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", norm="l2")),
        ("clf", MultinomialNB()),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = {}
print("=== Сравнение моделей (5-блочная кросс-валидация) ===")
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_f1[name] = scores.mean()
    print(f"{name:20s} | Macro F1: {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(cv_f1, key=cv_f1.get)
best_pipe = models[best_name]

# Подробная диагностика и визуализация главной ошибки (матрица ошибок на CV)
oof_pred = cross_val_predict(best_pipe, X_train, y_train, cv=cv, n_jobs=-1)
print(f"\n=== Отчёт по классам ({best_name}, CV) ===")
print(classification_report(y_train, oof_pred, target_names=genres, zero_division=0, digits=3))

ConfusionMatrixDisplay(confusion_matrix(y_train, oof_pred), display_labels=genres).plot(
    cmap="Blues", values_format="d", xticks_rotation=30
)
plt.title(f"Матрица ошибок: {best_name}")
plt.tight_layout(); plt.savefig("error_analysis.png", dpi=150); plt.close()

# === БЛОК 3. ФИНАЛЬНАЯ ПРОВЕРКА НА "СЕЙФЕ" И СОХРАНЕНИЕ ===
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
test_acc = accuracy_score(y_test, y_pred)

cm_off = confusion_matrix(y_test, y_pred)
np.fill_diagonal(cm_off, 0)
if cm_off.max() > 0:
    i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
    mistake = f"Чаще всего она путает «{ID_TO_GENRE[int(i)]}» и «{ID_TO_GENRE[int(j)]}»."
else:
    mistake = "Ошибок на тесте нет."

joblib.dump({"pipeline": best_pipe, "id_to_genre": ID_TO_GENRE}, "model.pkl")
print(f"\nЛучшая модель — {best_name}. Её ключевая метрика на новых данных — Macro F1 = {test_f1:.4f} (Accuracy = {test_acc:.4f}). {mistake}")
print("Сохранено: model.pkl и error_analysis.png")
