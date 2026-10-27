Галимов Кирилл Ильдарович

Сравниваю ItemCF, iALS, EASE, FM и ALS с измененным таргетом на [Amazon Reviews 2018, Video Games, 5-core](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/). Параметры выбираю по NDCG@10, дополнительно считаю Recall@10.

Код моделей находится в `models/`, а графики, результаты и выводы в [experiments.ipynb](experiments.ipynb).

Запуск:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/execute_notebook.py
```

Данные скачиваются автоматически, результаты берутся из `results/`. Чтобы обучить модели заново, нужно поставить `RETRAIN = True` в ноутбуке или запустить `python experiment.py`.
