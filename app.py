import os, re, joblib
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
_ARTIFACT = None


def predict_genre(description):
    """Проверяет ввод ('защита от дурака') и возвращает предсказанный жанр и Топ-3 строкой."""
    global _ARTIFACT
    if not isinstance(description, str) or len(description.strip()) < 15:
        return "Ошибка ввода: введите описание сюжета (минимум 15 символов)."
    if len(re.findall(r"[a-zA-Zа-яА-ЯёЁ]{2,}", description)) < 3:
        return "Ошибка ввода: введите осмысленный текст (минимум 3 слова), а не только цифры/символы."
    if len(description) > 10000:
        return "Ошибка ввода: текст слишком длинный (максимум 10 000 символов)."

    if _ARTIFACT is None:
        if not os.path.exists(MODEL_PATH):
            return "Ошибка: файл model.pkl не найден. Сначала запустите python train.py."
        _ARTIFACT = joblib.load(MODEL_PATH)

    probs = _ARTIFACT["pipeline"].predict_proba([description.strip()])[0]
    top3 = probs.argsort()[::-1][:3]
    id2g = _ARTIFACT["id_to_genre"]
    best = f"{id2g[int(top3[0])].upper()} ({probs[top3[0]] * 100:.1f}%)"
    details = ", ".join(f"{id2g[int(i)]} ({probs[i] * 100:.1f}%)" for i in top3)
    return f"Предсказанный жанр: {best}\nТоп-3: {details}"


iface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Textbox(lines=5, label="Описание фильма или книги", placeholder="Введите сюжет (от 3 слов)..."),
    outputs=gr.Textbox(lines=3, label="Результат"),
    title="GenreAI — Классификация жанра по описанию",
    description="Введите описание сюжета — модель определит жанр. Работает проверка некорректного ввода.",
    examples=[
        # Успешный пример (явный horror)
        ["A group of friends stays in an abandoned haunted cabin where a terrifying ghost hunts them at night."],
        # Пограничный пример (смесь comedy и drama)
        ["After losing his job, a clumsy father opens a funny bakery while dealing with family divorce and memories."],
    ],
)

if __name__ == "__main__":
    iface.launch()
