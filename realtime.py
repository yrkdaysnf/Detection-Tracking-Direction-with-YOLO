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
    if param == "show_trace": show_trace = state
    elif param == "show_centr": show_centr = state
    elif param == "show_bbox": show_bbox = state
    elif param == "show_direction": show_direction = state

# Состояния по умолчанию
show_trace = True
show_centr = True
show_bbox = True
show_direction = True

# Проверяем используется ли CUDA и подгружаем модель (выводим ее характеристики)
checks()
model = YOLO('yolov8n.pt')
model.fuse()

# Открываем видеопоток с камеры и настраиваем разрешение
cap = cv2.VideoCapture(0)
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

# Буфер точек положения центра объекта с фиксирвоанным размером 50 точек
trails = defaultdict(lambda: deque(maxlen=50))

# Переменные для подсчета FPS
fps_start_time, fps_frame_count, fps = time(), 0, 'FPS: 0'

# Основной цикл программы
while True:
    ret, frame = cap.read()
    if not ret:break

    # Добавляем рамку чтобы плашка не исчезала
    frame = cv2.copyMakeBorder(
                               src=frame, 
                               top=20,
                               bottom = 0,
                               left=0,
                               right=0, 
                               borderType=cv2.BORDER_CONSTANT, 
                               value=0
                              )
    
    # Обнуляем количество объектов
    obj = 0

    # Детекция и трекинг объекта средствами YOLO и (botsort/bytetrack)
    results = model.track(
                          source=frame,
                          iou=0.6,
                          conf=0.6,
                          persist=True,
                          verbose=False,
                          tracker='config\\bytetrack.yaml'
                         )

    # Вытаскиваем ограничивающие рамки, классы, уникальные номера и уверенности
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype(float)

        # Количество объектов
        obj = len(ids)

        # Обрабатываем информацию каждого объекта
        for box, cl, id, conf in zip(boxes, cls, ids, confs):
            # Координата центра, ширина и высота ограничивающей рамки
            cx,cy,w,h = box

            # Крайние точки для ограничивающей рамки
            x0, y0, x1, y1 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)

            # Добавляем координату центра объекта в массив
            trails[id].append((cx, cy))

            # Задаём цвета для красоты (Единый цвет у класса)
            r.seed(int(cl))
            color = (r.randint(100, 255), r.randint(50, 255), r.randint(25, 255))

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
                              thickness=2
                             )

            # Отображение направления движения (В траектории должно быть больше 10 точек)
            if show_direction and len(list(trails[id]))>10:
                # Инициализация списка для векторов перемещения
                movements = []

                # Получение массива центров объекта
                trail = list(trails[id])

                # Вычисление векторов перемещения по последним 10 точкам
                for i in range(len(trail)-10, len(trail)):
                    movements.append(np.array(trail[i]) - np.array(trail[i - 1]))

                # Усреднение векторов перемещения
                movement = np.mean(movements, axis=0)

                # Фильтруем тряску (Не показываем направление если движение малозначительное)
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
                # Подпись с уникальным номером, классом объекта и уверенностью в прцоентах
                text = f'Object #{id} ({results[0].names[cl]} - {int(conf*100)}%)'
                
                # Ограничивающая рамка
                cv2.rectangle(
                              img=frame,
                              pt1=(x0, y0),
                              pt2=(x1, y1),
                              color=color,
                              thickness=1
                             )
                
                # Плашка под текст
                frame[
                      (y0 - 20):(y0), 
                      (x0):(x0 + cv2.getTextSize(text, cv2.FONT_ITALIC, 0.5, 1)[0][0])
                     ] = color
                           
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
                # Подпись с координатой x, y центра объекта
                text = f'{cx, cy}'
                
                # Плашка под текст
                frame[
                      (cy - 30):(cy - 10), 
                      (cx + 10):(cx + cv2.getTextSize(text, cv2.FONT_ITALIC, 0.5, 1)[0][0] + 10)
                     ] = color

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
                
                # Рисуем точку центра объекта
                cv2.circle(frame, (cx, cy), 4, color, -1)        

    # Добавляем кадр
    fps_frame_count += 1

    # Подсчитываем кадры в секунду
    if time() - fps_start_time >= 1:
        # Рассчитать FPS
        fps = f'FPS: {int(fps_frame_count / (time() - fps_start_time))}'
        
        # Сбросить счетчики
        fps_start_time = time()
        fps_frame_count = 0
    
    # Техническая информация
    text=f'{fps} - Objects in frame: {obj}'

    # Обрезаем изображение (убираем ранее добавленную рамку)
    frame = frame[20:740, 0:1280]
        
    # Добавляем плашку под техническую информацию
    frame [5:25, 5:cv2.getTextSize(text, cv2.FONT_ITALIC, 0.5, 1)[0][0]+15] = 0
    
    # Выводим техническую информацию (Кадры в секунду и количество объектов)
    cv2.putText(
                img=frame,
                text=text,
                org=(10, 20),
                fontFace=cv2.FONT_ITALIC,
                fontScale=0.5,
                color=(255, 255, 255),
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