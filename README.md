# Lighthouse

Оснавная инструкция представлена в [README оригинального репозитория Lighthouse](https://github.com/line/lighthouse/blob/main/README.md)

В форке добавлена следующая функциональность:
- Модель UVCOM с модификацией архитекутры [Length-aware decoder](https://arxiv.org/html/2412.20816v3). Для обучения, дообучения и валидации модели необходимо использовать команды из основной инструкции, при этом для аргумента --model установить значение uvcom_lad. В файле congins/model/uvcom_lad.yml можно настроить параметры количество классов длительности и их нормализованные границы. Для подбора числа классов и границ для набора данных можно использовать скрипты training/standalone_eval/cumulative_map.py и training/standalone_eval/find_treshholds.py. Визуализировать полученные разбиения можно через подучу полученных json файлов в скрипт utils/visualize.py.
- Реализация оригинального и модифицированного алгоритма аугментации данных [MomentMix](https://arxiv.org/html/2412.20816v3). Для использования алгоритма во время обучения или дообучения необходимо добавить к основной строке запуска флаг --moment_mix (для выбора немодифицированного алгоритма необходимо в скрипте training/train.py указать параметр moment_mix_num_bg_candidates = 1, так же можно настроить и другие параметры алгоритма), например:
```
python training/train.py --model uvcom_lad --dataset castella --feature clap --moment_mix
```
- Скрипт для пеерфразирования запросов моделью DeepSeek V3 utils/query_aug.py
- Для Audio Corpus Moment Retrieval в режиме инференса при запуске через training/evaluate.py необходимо указать размер корпуса --num_distractors (выбираются случайным образом из выбранного набора данных, seed зафиксирован). Так же можно настроить параметры alpha и количество отбираемых для второго этапа локализации кандидатов: --alpha и --top_k. Например: 
```
python training/evaluate.py --model uvcom --dataset castella --feature clap --split val --model_path best.ckpt --eval_path data/castella/castella_test_release.jsonl --num_distractors 100 --alpha 0.1 --top_k 3
```
- Разбиение посчитанных метрик по диапозонам длительности 0-10, 10-20, ..., 90+ при вызове training/evaluate.py. Полученные результаты можно визуализировать utils/histogram_metric_by_moment_length.py.
- Визуализация loss и метрик utils/plot_loss.py utils/plot_val_metrics.py из train.json и val.json файлов, полученных при обучении модели.

[Чекпоинты для моделей uvcom и uvcom_lad ](https://drive.google.com/drive/folders/1PwyUKxvgZV5ENpXK4HbWiwUbURJecqSe?usp=sharing) 

Метрики на CASTELLA:

| Method | R@1, 0.5 | R@1, 0.7 | mAP@avg | mAP@0.5 | mAP@0.75 |
|--------|----------|----------|---------|---------|----------|
| UVCOM base | 31.7 | 20.3 | 15.9 | 28.4 | 15.2 |
| UVCOM LAD 3 class [0.05, 0.2] + MomentMix * + Query | 34.7 (+9.4%) | 22.1 (+8.8%) | 18.0 (+13.2%) | 30.6 (+7.7%) | 17.4 (+17.1%) |

Метрики на UnAV-100 subset:

| Method | R@1, 0.5 | R@1, 0.7 | mAP@avg | mAP@0.5 | mAP@0.75 |
|--------|----------|----------|---------|---------|----------|
| UVCOM base | 54.0 | 46.0 | 41.6 | 61.5 | 44.5 |
| UVCOM LAD 3 class [0.05, 0.2] + MomentMix * + Query | 57.0 (+5.6%) | 47.0 (+2.2%) | 45.7 (+9.9%) | 65.6 (+6.7%) | 48.4 (+8.8%) |