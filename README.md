# Detection-Tracking-Direction-with-YOLO

Данный проект является программной частью моего дипломного проекта бакалавра СПБГЭТУ "ЛЭТИ"

>Тема проекта: _Создание системы определения направления движения и положения объекта в пространстве._ 

### Для запуска:
1) Установите __virtualenv__:

```
pip3 install virtualenv
``` 

2) Создайте виртуальное окружение: 

```
python3 -m virtualenv .venv
```

3) Активируйте виртуальное окружение:
<details><summary>Linux</summary>
<pre><code>source .venv/bin/activate</code></pre>
</details>
<details><summary>Windows</summary>
<pre><code>.venv/Scripts/activate</code></pre>
</details>

4) Установите библиотеку __ultralytics__, в ней всё необходимое для запуска:

```
pip3 install ultralytics
```

5) Отредактируйте код на своё усмотрение.

6) Запустите интересующий скрипт - `realtime.py` или `video.py`:

```
python3 (название скрипта) # realtime.py или video.py
```

7) Если вы получили сообщение по типу:
```
Ultralytics YOLOv8.1.47 🚀 Python-3.8.10 torch-2.2.2+cu121 CUDA:0 (NVIDIA GeForce GTX 1660, 5928MiB)
Setup complete ✅ (6 CPUs, 31.3 GB RAM, 65.2/109.5 GB disk)
YOLOv8m summary (fused): 218 layers, 25886080 parameters, 0 gradients, 78.9 GFLOPs
```
<p align='center'> и появилось подобное окно: </p>

<p align='center'>
<img src=assets/window.png/>
</p>

<p align='center'> Поздравляю! Всё заработало! </p>

### Что-то не так?
>⚠️ Если при запуске `realtime.py`, в консоли вы не нашли название своей видеокарты, то вероятно потребуется перезапуск системы, во всех других случаях, попробуйте установить __pytorch__ отдельно, для конкретно вашей системы с [оффициального сайта](https://pytorch.org/get-started/locally/).

<p align="center">
<img src=assets/pytorch.png />
</p>

>⚠️ Убедитесь что виртуальное окружение активировано
