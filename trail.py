import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
import random as r
import numpy as np


# Обработка состояния чекбокса
def on_checkbox_change(state, param):
    global show_trace, show_centr, show_bbox, show_predict
    if param == "show_trace":
        show_trace = state
    elif param == "show_centr":
        show_centr = state
    elif param == "show_bbox":
        show_bbox = state
    elif param == "show_predict":
        show_predict = state

# Состояния по умолчанию
show_trace = False
show_centr = False
show_bbox = False
show_predict = True

# Подгружаем модель
model = YOLO('yolov8m.pt')
model.fuse()

# Открываем видеопоток с камеры и настраиваем разрешение
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Буфер точек положения центра объекта
trail_queues = defaultdict(lambda: deque(maxlen=10))

# Создаем окно
cv2.namedWindow("frame")

# Добавляем чекбоксы
cv2.createTrackbar("Bbox", "frame", int(show_bbox), 1, lambda state, 
                   param="show_bbox": on_checkbox_change(state, param))
cv2.createTrackbar("Center", "frame", int(show_centr), 1, lambda state, 
                   param="show_centr": on_checkbox_change(state, param))
cv2.createTrackbar("Trace", "frame", int(show_trace), 1, lambda state, 
                   param="show_trace": on_checkbox_change(state, param))
cv2.createTrackbar("Predict", "frame", int(show_predict), 1, lambda state, 
                   param="show_predict": on_checkbox_change(state, param))


# Основной цикл программы
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Детекция и трекинг объекта средствами YOLO и (botsort/bytetrack)
    results = model.track(frame,
                          iou=0.6,
                          conf=0.6,
                          persist=True,
                          verbose=False,
                          tracker='botsort.yaml')

    # Пакуем полученную информацию от YOLO
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        conf = results[0].boxes.conf.cpu().numpy().astype(float)

        # Обрабатываем информацию каждого объекта
        for box, cl, id, con in zip(boxes, cls, ids, conf):
            # Координаты для Bounding Box
            x0,y0,x1,y1 = box

            # Центр объекта
            cx, cy = int(x0+(x1-x0)/2), int(y0+(y1-y0)/2)

            # Добавляем точку в массив
            trail_queues[id].append((cx, cy))

            # Задаём цвета для красоты
            r.seed(int(cl))
            color = (r.randint(100, 200),
                     r.randint(50, 200),
                     r.randint(25, 200))
            
            # Отображение "Окружающего прямоугольника"
            if show_bbox:
                # Рамка
                cv2.rectangle(img=frame,
                              pt1=(x0, y0), 
                              pt2=(x1, y1), 
                              color=color, 
                              thickness=1)
                
                # Имя объекта и точность
                text = f'{model.model.names[cl]} - {int(con*100)}%'
                
                # Плашка под текст
                frame[(y0-20):y0, x0:x0+(len(text)*10)] = color

                # Добавляем текст
                cv2.putText(img=frame,
                            text=text,
                            org=(x0, y0-5),
                            fontFace=cv2.FONT_ITALIC,
                            fontScale=0.5,
                            color=(0, 0, 0),
                            thickness=1)

            # Отображение центра объекта
            if show_centr:cv2.circle(frame, (cx, cy), 5, color, -1)

            # Отображение траектории
            if show_trace:
                for i in range(1, len(trail_queues[id])):
                    cv2.line(frame, trail_queues[id][i-1], trail_queues[id][i], color, 1)
            
            # Отображение направления движения
            if show_predict and len(list(trail_queues[id]))>=5:
                # Инициализация списка для векторов перемещения
                movement_vectors = []

                # Получение массива центров объекта
                trail = list(trail_queues[id])

                # Вычисление векторов перемещения
                for i in range(1, len(trail)):
                    prev_center = np.array(trail[i - 1])
                    current_center = np.array(trail[i])
                    movement_vector = current_center - prev_center
                    movement_vectors.append(movement_vector)

                # Усреднение векторов перемещения
                average_movement_vector = np.mean(movement_vectors, axis=0)

                # Определение координат конца стрелки
                point = tuple((np.array(trail[-1]) + 10 * average_movement_vector).astype(int))

                # Нарисовать стрелку, указывающую на направление движения
                cv2.arrowedLine(frame, trail[-1], point, color, 3)

    # Показываем что получилось
    cv2.imshow('frame', frame)

    # Выход по нажатию "Esc"
    if cv2.waitKey(1) == 27:
        break

# Высвобождаем ресурсы камеры
cap.release()

# Закрываем все окна
cv2.destroyAllWindows()