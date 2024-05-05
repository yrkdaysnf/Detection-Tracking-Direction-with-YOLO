# Стандартные модули
import random as r
from collections import defaultdict, deque
from tqdm import tqdm
from datetime import datetime 
# Сторонние модули
from ultralytics import YOLO, checks
import cv2
import numpy as np


# Состояния по умолчанию
show_trace = True
show_centr = True
show_bbox = True
show_direction = True

# Проверяем используется ли CUDA и подгружаем модель (выводим ее характеристики)
checks()
model = YOLO('yolov8x.pt')
model.fuse()

# Открываем видеопоток и создаем объект для сохранения
file = 'test.mp4'
cap = cv2.VideoCapture(f'input\\{file}')
fps = float(cap.get(cv2.CAP_PROP_FPS))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = f'output\\{file.split(".")[0]}_{datetime.now().strftime("%H%M%S")}.mp4'
out = cv2.VideoWriter(
                      filename=output_path, 
                      fourcc=fourcc, 
                      fps=fps, 
                      frameSize=(1920, 1080)
                     )

# Буфер точек положения центра объекта
trails = defaultdict(lambda: deque(maxlen=50))

# Основной цикл программы
for _ in tqdm(range(frames-1), unit=' frames'):
    _, frame = cap.read()

    # Меняем размер кадра
    

    # Добавляем рамку чтобы плашка не исчезала
    frame = cv2.copyMakeBorder(
                               src=cv2.resize(frame, (1920, 1080)), 
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
        obj = len(ids)

        # Обрабатываем информацию каждого объекта
        for box, cl, id, conf in zip(boxes, cls, ids, confs):
            # Координата центра, ширина и высота
            cx,cy,w,h = box

            # Крайние точки для ограничивающей рамки
            x0, y0, x1, y1 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)

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
                              thickness=2
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
                # Уникальный номер, класс объекта и уверенность
                text = f'Object #{id} ({results[0].names[cl]} - {int(conf*100)}%)'
                
                # Рамка
                cv2.rectangle(
                              img=frame,
                              pt1=(x0, y0),
                              pt2=(x1, y1),
                              color=color,
                              thickness=1
                             )
                
                # Плашка под текст
                frame[
                      (y0 - 20):y0, 
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
                # Координаты x, y центра объекта
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
                
                # Рисуем точку (центр)
                cv2.circle(frame, (cx, cy), 4, color, -1)
        
    # Обрезаем изображение (убираем ранее добавленную рамку)
    frame = frame[20:1100, 0:1980]

    # Техническая информация
    text=f'Objects in frame: {obj}'
        
    # Добавляем плашку под техническую информацию
    frame [5:40, 5:cv2.getTextSize(text, cv2.FONT_ITALIC, 1, 1)[0][0]+15] = 0
    
    # Выводим техническую информацию
    cv2.putText(
                img=frame,
                text=text,
                org=(10, 30),
                fontFace=cv2.FONT_ITALIC,
                fontScale=1,
                color=(255, 255, 255),
                thickness=1
               )

    out.write(frame)

# Высвобождаем ресурсы
cap.release()
out.release()

print('Видео успешно сохранено!')

# Закрываем все окна
cv2.destroyAllWindows()