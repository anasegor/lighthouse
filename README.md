# Lighthouse

![Основное README](https://github.com/line/lighthouse/blob/main/README.md)

В форке добавлена следующая функциональность:
- Модель UVCOM с модификацией архитекутры [Length-aware decoder](https://arxiv.org/html/2412.20816v3). Для обучения, дообучения и валидации модели необходимо использовать команды из основной инструкцию, для флага --model использовать значение uvcom_lad. [Чекпоинты](https://drive.google.com/drive/folders/1PwyUKxvgZV5ENpXK4HbWiwUbURJecqSe?usp=sharing) 
- Реализация оригинального и модифицированного алгоритма аугментации данных [MomentMix](https://arxiv.org/html/2412.20816v3). Для использования алгоритма во время обучения или дообучения необходимо добавить к основной строке запуска флаг --moment_mix (для выбора немодифицированного алгоритма необходимо в скрипте training/train.py указать параметр moment_mix_num_bg_candidates = 1, так же можно настроить и другие параметры алгоритма), например:
```
python training/train.py --model uvcom_lad --dataset castella --feature clap --moment_mix
```
- Для Audio Corpus Moment Retrieval в режиме инференса при запуске через training/evaluate.py необходимо указать размер корпуса --num_distractors (выбираются случайным образом из выбранного набора данных, seed зафиксирован). Так же можно настроить параметры alpha и кол-во кандидатов во второй этап локализации: --alpha и --top_k. Например: 
```
python training/evaluate.py --model uvcom --dataset castella --feature clap --split val --model_path best.ckpt --eval_path data/castella/castella_test_release.jsonl --num_distractors 100 --alpha 0.1 --top_k 3
```