import random as r # Случайный цвет для класса объектов
from ultralytics import YOLO # Нейросеть YOLO
import cv2 # Чтение видеопотока и визуализация результатов YOLO


# Функция для визуализации ограничивающей рамки объекта
def draw_bbox(frame, name, conf, color, x0, y0, x1, y1):
    # Подпись с классом объекта и уверенностью в процентах
    text = f'{name} - {int(conf*100)}%' 
    
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

cap = cv2.VideoCapture(0) # Открываем видеопоток с камеры
#Настраиваем разрешение
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

model = YOLO('models\\yolov8m.pt') # Подгружаем модель YOLOv8m
model.fuse() # Объединение слоев для оптимизации вывода.
names = model.model.names # Вытаскиваем имена классов

# Основной цикл программы
while True:
    ret, frame = cap.read() # ret - True если кадр(frame) существует
    if not ret:break # Выход из цикла если кадра нет.

    # Добавляем рамку чтобы плашка в верхней части кадра не исчезала
    frame = cv2.copyMakeBorder(src=frame, top=20, bottom = 0,left=0, right=0, 
                               borderType=cv2.BORDER_CONSTANT, value=0)
    
    # Детекция объекта средствами YOLO
    results = model.predict(source=frame, iou=0.6, conf=0.6, verbose=False)

    # Извлекаем ограничивающие рамки, классы и уверенности
    if results[0].boxes.cls is not None:
        boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype(float)

        # Последовательно обрабатываем информацию каждого объекта
        for box, cl, conf in zip(boxes, cls, confs):
            name = names[cl] # Имя класса объекта
            cx,cy,w,h = box # Координата центра, ширина и высота ограничивающей рамки
            
            # Крайние точки для ограничивающей рамки
            x0, y0, = int(cx - w/2), int(cy - h/2)
            x1, y1 = int(cx + w/2), int(cy + h/2)

            # Задаём цвета для красоты (Единый цвет у класса)
            r.seed(int(cl))
            color = (r.randint(150, 255), r.randint(150, 255), r.randint(150, 255))

            # Отображение ограничивающей рамки
            draw_bbox(frame, name, conf, color, x0, y0, x1, y1)

            # Отображение центра объекта с координатой
            draw_centr(frame, color, cx, cy)

    frame = frame[20:740, 0:1280] # Обрезаем изображение (убираем ранее добавленную рамку)
    cv2.imshow('Detection + Tracking + Direction', frame) # Выводим результат
    if cv2.waitKey(1) == 27:break # Выход по нажатию "Esc"

cap.release() # Высвобождаем ресурсы камеры
cv2.destroyAllWindows() # Закрываем все окна