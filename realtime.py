# Стандартные модули
import random as r
from collections import defaultdict, deque
from time import time
# Сторонние модули
from ultralytics import YOLO, checks
import cv2
import numpy as np


# Обработка состояния чекбокса
def checkbox(state, param):
    global show_trace, show_centr, show_bbox, show_direction
    if param == "show_trace":
        show_trace = state
    elif param == "show_centr":
        show_centr = state
    elif param == "show_bbox":
        show_bbox = state
    elif param == "show_direction":
        show_direction = state

# Состояния по умолчанию
show_trace = True
show_centr = True
show_bbox = True
show_direction = True

# Проверяем используется ли CUDA и подгружаем модель (выводим ее характеристики)
checks()
model = YOLO('yolov8m.pt')
model.fuse()

# Открываем видеопоток с камеры и настраиваем разрешение
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Создаем окно
cv2.namedWindow('Detection + Tracking + Direction')

# Добавляем чекбоксы
cv2.createTrackbar(
                    "Bbox", 'Detection + Tracking + Direction', int(show_bbox), 1,
                    lambda state, param="show_bbox": checkbox(state, param)
                    )

cv2.createTrackbar(
                    "Center", 'Detection + Tracking + Direction', int(show_centr), 1,
                    lambda state, param="show_centr": checkbox(state, param)
                    )

cv2.createTrackbar(
                    "Trace", 'Detection + Tracking + Direction', int(show_trace), 1, 
                    lambda state, param="show_trace": checkbox(state, param)
                    )

cv2.createTrackbar(
                    "Direction", 'Detection + Tracking + Direction', int(show_direction), 1, 
                    lambda state, param="show_direction": checkbox(state, param)
                    )

# Буфер точек положения центра объекта
trails = defaultdict(lambda: deque(maxlen=50))

# Переменные для подсчета FPS
fps_start_time, fps_frame_count, fps = time(), 0, 'Please wait...'

# Основной цикл программы
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Добавляем рамку чтобы плашка не исчезала
    frame = cv2.copyMakeBorder(frame, 20, 20, 5, 5, cv2.BORDER_CONSTANT, value=(240,240,240))
    
    # Детекция и трекинг объекта средствами YOLO и (botsort/bytetrack)
    results = model.track(
                        frame,
                        iou=0.6,
                        conf=0.6,
                        persist=True,
                        verbose=False,
                        tracker='config\\bytetrack.yaml'
                        )

    # Вытаскиваем ограничивающуе рамки, центры, классы, уникальные номера и уверенности
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        centres = results[0].boxes.xywh.cpu().numpy().astype(int)
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype(float)

        # Обрабатываем информацию каждого объекта
        for box, centr, cl, id, conf in zip(boxes, centres, cls, ids, confs):
            # Координаты для ограничивающей рамки
            x0,y0,x1,y1 = box

            # Центр объекта
            cx, cy, _, _ = centr

            # Добавляем координату центра объекта в массив
            trails[id].append((cx, cy))

            # Задаём цвета для красоты (Единый цвет у класса)
            r.seed(int(cl))
            color = (r.randint(100, 200), r.randint(50, 200), r.randint(25, 200))

            # Отображение траектории
            if show_trace:
                # Конвертируем массив точек для функции cv2.polylines()
                points = np.hstack(trails[id]).astype(np.int32).reshape((-1, 1, 2))

                # Рисуем траекторию
                cv2.polylines(
                            img=frame,
                            pts=[points],
                            isClosed=False,
                            color=color,
                            thickness=3
                            )

            # Отображение направления движения
            if show_direction and len(list(trails[id]))>10:
                # Инициализация списка для векторов перемещения
                movements = []

                # Получение массива центров объекта
                trail = list(trails[id])

                # Вычисление векторов перемещения
                for i in range(len(trail)-10, len(trail)):
                    movements.append(np.array(trail[i]) - np.array(trail[i - 1]))

                # Усреднение векторов перемещения
                movement = np.mean(movements, axis=0)

                # Фильтруем тряску
                if abs(movement).astype(float)[0] > 0.3 or abs(movement).astype(float)[1] > 0.3:
                    # Определение координат конца стрелки
                    point = tuple((np.array(trail[-1]) + 10 * movement).astype(int))

                    # Нарисовать стрелку, указывающую на направление движения
                    cv2.arrowedLine(
                                    img=frame,
                                    pt1=trail[-1],
                                    pt2=point,
                                    color=color,
                                    thickness=2
                                    )
            
            # Отображение ограничивающей рамки
            if show_bbox:
                # Имя объекта и точность
                text = f'{results[0].names[cl]} - {int(conf*100)}%'
                
                # Рамка
                cv2.rectangle(
                            img=frame,
                            pt1=(x0, y0),
                            pt2=(x1, y1),
                            color=color,
                            thickness=2
                            )
                
                # Плашка под текст
                frame[(y0 - 20):y0, x0:x0 + (len(text) * 10)] = color
                           
                # Добавляем текст
                cv2.putText(
                            img=frame,
                            text=text,
                            org=(x0, y0-5),
                            fontFace=cv2.FONT_ITALIC,
                            fontScale=0.5,
                            color=0,
                            thickness=1
                            )

            # Отображение центра объекта с координатой
            if show_centr:
                # Координаты x, y центра объекта
                text = f'{cx, cy}'
                
                # Плашка под текст
                frame[(cy - 30):(cy - 10), (cx + 10):(cx + (len(text) * 10))] = color

                # Добавляем текст
                cv2.putText(
                            img=frame,
                            text=text,
                            org=(cx + 10, cy - 15),
                            fontFace=cv2.FONT_ITALIC,
                            fontScale=0.5,
                            color=0,
                            thickness=1
                            )
                
                # Рисуем точку (центр)
                cv2.circle(frame, (cx, cy), 4, color, -1)

        # Выводим кол-во объектов
        cv2.putText(
                    img=frame,
                    text=f'Objects in frame: {len(results[0].boxes.id)}',
                    org=(5, 755),
                    fontFace=cv2.FONT_ITALIC, 
                    fontScale=0.5, 
                    color=0,
                    thickness= 1
                    )

    #Добавляем кадр
    fps_frame_count += 1

    # Вывод каждую 1 секунду
    if time() - fps_start_time >= 1:
        # Рассчитать FPS
        fps = f'FPS: {int(fps_frame_count / (time() - fps_start_time))}'
        
        # Сбросить счетчики
        fps_start_time = time()
        fps_frame_count = 0
    
    # Выводим кадры в секунду
    cv2.putText(
                img=frame,
                text=fps, 
                org=(5, 15),
                fontFace=cv2.FONT_ITALIC,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1
                )

    # Показываем что получилось
    cv2.imshow('Detection + Tracking + Direction', frame)

    # Выход по нажатию "Esc"
    if cv2.waitKey(1) == 27:break

# Высвобождаем ресурсы камеры
cap.release()

# Закрываем все окна
cv2.destroyAllWindows()