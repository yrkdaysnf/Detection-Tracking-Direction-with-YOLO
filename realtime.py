from os import path as p # Создание рабочих путей для UNIX и Windows 
import random as r # Случайный цвет для класса объектов
from collections import defaultdict, deque # Фиксированный массив для трекинга
from time import time # Расчет кадров в секунду
from ultralytics import YOLO, checks # Нейросеть YOLO и проверка доступности GPU
import cv2 # Визуализация результатов YOLO
import numpy as np # Преобразования матриц и вычисление вектора направленности


# Функция для обработки состояния чекбокса
def checkbox(state, param):
    global show_trace, show_centr, show_bbox, show_direction
    if param == "show_trace": show_trace = state
    elif param == "show_centr": show_centr = state
    elif param == "show_bbox": show_bbox = state
    elif param == "show_direction": show_direction = state

# Функция для визуализации направления движения объекта
def draw_direction(frame, trail, color):
    movements = [] # Инициализация списка для векторов перемещения
    trail = list(trail) # Получение массива центров объекта

    # Вычисление векторов перемещения по последним 10 точкам
    for i in range(len(trail)-10, len(trail)):
        movements.append(np.array(trail[i]) - np.array(trail[i - 1]))

    movement = np.mean(movements, axis=0) # Усреднение векторов перемещения

    # Фильтруем тряску (Не показываем направление если движение незначительное)
    if abs(movement).astype(float)[0] > 0.3 or abs(movement).astype(float)[1] > 0.3:
        # Определение координат конца стрелки
        point = tuple((np.array(trail[-1]) + 10 * movement).astype(int))

        # Нарисовать стрелку, указывающую на направление движения
        cv2.arrowedLine(img=frame, pt1=trail[-1], pt2=point, color=color, thickness=2)

# Функция для визуализации траектории объекта
def draw_trace(frame, trail, color):
    # Конвертируем массив точек для функции cv2.polylines()
    points = np.hstack(trail).astype(np.int32).reshape((-1, 1, 2))

    # Рисуем траекторию
    cv2.polylines(img=frame, pts=[points], isClosed=False, color=color, thickness=2)

# Функция для визуализации ограничивающей рамки объекта
def draw_bbox(frame, id, name, conf, color, x0, y0, x1, y1):
    # Подпись с уникальным номером, классом объекта и уверенностью в процентах
    text = f'#{id} ({name} - {int(conf*100)}%)'
    
    # Ограничивающая рамка
    cv2.rectangle(img=frame, pt1=(x0, y0), pt2=(x1, y1), color=color, thickness=1)
    
    # Плашка под текст
    frame[(y0 - 20):(y0), 
          (x0):(x0 + cv2.getTextSize(text, cv2.FONT_ITALIC, 0.5, 1)[0][0])] = color
                
    # Добавляем текст
    cv2.putText(img=frame, text=text, org=(x0, y0-5), 
                fontFace=cv2.FONT_ITALIC, fontScale=0.5, color=0, thickness=1)

# Функция для визуализации центра объекта
def draw_centr(frame, color, cx, cy):
    text = f'{cx, cy}' # Подпись с координатой x, y центра объекта
    
    # Плашка под текст
    frame[(cy - 30):(cy - 10),
          (cx + 10):(cx + cv2.getTextSize(text, cv2.FONT_ITALIC, 0.5, 1)[0][0] + 10)] = color

    # Добавляем текст
    cv2.putText(img=frame,text=text,org=(cx + 10, cy - 15),
                fontFace=cv2.FONT_ITALIC, fontScale=0.5, color=0, thickness=1)
    
    cv2.circle(frame, (cx, cy), 4, color, -1) # Рисуем точку центра объекта

# Переменные отвечающие за показ визуализации
show_trace = True
show_direction = True
show_bbox = True
show_centr = True

checks() # Проверяем используется ли CUDA
name_model = 'yolov8m' # Выбор модели
name_tracker = 'botsort' # Выбор трекера
model = YOLO(p.join('models', f'{name_model}.pt')) # Подгружаем модель YOLOv8m
model.fuse() # Объединение слоев для оптимизации вывода
names = model.model.names # Вытаскиваем имена классов

cap = cv2.VideoCapture(0) # Открываем видеопоток с камеры
#Настраиваем разрешение
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

cv2.namedWindow('Detection + Tracking + Direction') # Создаем окно

# Добавляем ползунки в ранее созданное окно (Поскольку нет кнопок, использую ползунки)
cv2.createTrackbar("Bbox", 'Detection + Tracking + Direction', int(show_bbox), 1,
                   lambda state, param="show_bbox": checkbox(state, param))
cv2.createTrackbar("Center", 'Detection + Tracking + Direction', int(show_centr), 1,
                   lambda state, param="show_centr": checkbox(state, param))
cv2.createTrackbar("Trace", 'Detection + Tracking + Direction', int(show_trace), 1, 
                   lambda state, param="show_trace": checkbox(state, param))
cv2.createTrackbar("Direction", 'Detection + Tracking + Direction', int(show_direction), 1, 
                   lambda state, param="show_direction": checkbox(state, param))

# Массив точек положения центра объекта с фиксированным размером = 50
trails = defaultdict(lambda: deque(maxlen=50))

# Переменные для подсчета кадров в секунду
start_time, frame_count, fps = time(), 0, 'FPS: 0'

# Основной цикл программы
while True:
    ret, frame = cap.read() # ret - True если кадр(frame) существует
    if not ret:break # Выход из цикла если кадра нет.

    # Добавляем рамку чтобы плашка не исчезала
    frame = cv2.copyMakeBorder(src=frame, top=20, bottom = 0,left=0, right=0, 
                               borderType=cv2.BORDER_CONSTANT, value=0)
    
    obj = 0 # Обнуляем количество объектов

    # Детекция и трекинг объекта средствами YOLO
    results = model.track(source=frame, iou=0.6, conf=0.6, verbose=False,
                          persist=True, tracker=p.join('config', f'{name_tracker}.yaml'))

    # Вытаскиваем ограничивающие рамки, классы, уникальные номера и уверенности
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype(float)

        obj = len(ids) # Количество объектов

        # Обрабатываем информацию каждого объекта
        for box, cl, id, conf in zip(boxes, cls, ids, confs):
            name = names[cl] # Имя класса объекта
            cx,cy,w,h = box # Координата центра, ширина и высота ограничивающей рамки
            
            # Крайние точки для ограничивающей рамки
            x0, y0, = int(cx - w/2), int(cy - h/2)
            x1, y1 = int(cx + w/2), int(cy + h/2)

            # Задаём цвета для красоты (Единый цвет у класса)
            r.seed(int(cl))
            color = (r.randint(150, 255), r.randint(150, 255), r.randint(150, 255))

            trails[id].append((cx, cy)) # Добавляем координату центра объекта в массив

            # Отображение траектории
            if show_trace: draw_trace(frame, trails[id], color)

            # Отображение направления движения (В траектории должно быть больше 10 точек)
            if show_direction and len(list(trails[id]))>10:draw_direction(frame, trails[id], color)
                
            # Отображение ограничивающей рамки
            if show_bbox:draw_bbox(frame, id, name, conf, color, x0, y0, x1, y1)

            # Отображение центра объекта с координатой
            if show_centr:draw_centr(frame, color, cx, cy)

    frame_count += 1 # Добавляем кадр

    # Подсчитываем кадры в секунду
    if time() - start_time >= 1:
        fps = f'FPS: {int(frame_count / (time() - start_time))}' # Рассчитываем FPS
        
        # Сбросить счетчики
        start_time = time()
        frame_count = 0
    
    text=f'{fps} - Objects in frame: {obj}' # Техническая информация
    frame = frame[20:740, 0:1280] # Обрезаем изображение (убираем ранее добавленную рамку)
        
    # Добавляем плашку под техническую информацию
    frame [5:25, 5:cv2.getTextSize(text, cv2.FONT_ITALIC, 0.5, 1)[0][0]+15] = 0
    
    # Выводим техническую информацию (Кадры в секунду и количество объектов)
    cv2.putText(img=frame, text=text, org=(10, 20), fontFace=cv2.FONT_ITALIC,
                fontScale=0.5, color=(255, 255, 255), thickness=1)

    cv2.imshow('Detection + Tracking + Direction', frame) # Выводим результат
    if cv2.waitKey(1) == 27:break # Выход по нажатию "Esc"

cap.release() # Высвобождаем ресурсы камеры
cv2.destroyAllWindows() # Закрываем все окна