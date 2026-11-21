import os, sys, importlib

# Добавляем корень проекта в sys.path, чтобы импорт работал при запуске 'pytest tests/'
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tests" else HERE
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

app = None
for mod_name in ("app", "app-v4", "app-v3"):
    if os.path.exists(os.path.join(ROOT, f"{mod_name}.py")):
        app = importlib.import_module(mod_name)
        break
assert app is not None, "Не найден файл app.py в корне проекта."


def test_predict_runs():
    """Тест 1: основная функция загружает модель и не падает на корректном примере."""
    res = app.predict_genre("A brave police agent fights a dangerous gang in a city full of explosions and chases.")
    assert res and "Ошибка" not in res


def test_predict_format_and_foolproofing():
    """Тест 2: возвращается строка в верном формате, а пустой/мусорный ввод перехватывается."""
    res = app.predict_genre("A terrifying ghost haunts an old dark house while teenagers try to survive the night.")
    assert isinstance(res, str) and "Предсказанный жанр:" in res

    assert "Ошибка ввода" in app.predict_genre("")
    assert "Ошибка ввода" in app.predict_genre("12345 67890 !!!")


def test_web_app_ready():
    """Тест 3: веб-интерфейс Gradio создаётся без ошибок."""
    assert app.iface is not None
