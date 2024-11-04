import os
import sys
import json 
import codecs

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import get_model
from utils.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from utils.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
from utils.FloorplanToBlenderLib import *
discrete_cmap()
from utils.post_prosessing import split_prediction, get_polygons, split_validation
from mpl_toolkits.axes_grid1 import AxesGrid

img_path = "input/image.png"


rot = RotateNTurns()
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath",
                "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience", "Toilet", "Sink",
                "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

# Загружаем модель
model = get_model('hg_furukawa_original', 51)
n_classes = 44
split = [21, 12, 11]
model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

# Загружаем контрольную точку
checkpoint = torch.load('model_best_val_loss_var.pkl', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])
model.eval()  # Устанавливаем режим оценки

# Чтение изображения
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Исправление цветовых каналов

# Преобразование изображения в диапазон (-1, 1)
img = 2 * (img / 255.0) - 1

# Перемещение осей (h,w,3)--->(3,h,w) для соответствия входным данным модели
img = np.moveaxis(img, -1, 0)

# Преобразуем в тензор PyTorch и перемещаем на CPU
img = torch.tensor([img.astype(np.float32)])

n_rooms = 12
n_icons = 11

with torch.no_grad():
    # Проверяем, четные или нечетные размеры изображения
    size_check = np.array([img.shape[2], img.shape[3]]) % 2

    height = img.shape[2] - size_check[0]
    width = img.shape[3] - size_check[1]

    img_size = (height, width)

    # Повороты для аугментации
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    pred_count = len(rotations)
    prediction = torch.zeros([pred_count, n_classes, height, width])

    for i, r in enumerate(rotations):
        forward, back = r
        # Поворачиваем изображение
        rot_image = rot(img, 'tensor', forward)
        pred = model(rot_image)  # Прогноз модели
        # Поворачиваем обратно предсказание
        pred = rot(pred, 'tensor', back)
        # Исправляем карты предсказаний
        pred = rot(pred, 'points', back)
        # Приводим размер предсказания к исходному
        pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
        # Добавляем предсказание к выходному массиву
        prediction[i] = pred[0]

    # Среднее по предсказаниям
    prediction = torch.mean(prediction, 0, True)

# Предсказание комнат
rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
rooms_pred = np.argmax(rooms_pred, axis=0)

# Предсказание иконок
icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
icons_pred = np.argmax(icons_pred, axis=0)

# Постобработка
heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

# Выделение стен
wall_polygon_numbers = [i for i, j in enumerate(types) if j['type'] == 'wall']
boxes = []
for i, j in enumerate(polygons):
    if i in wall_polygon_numbers:
        temp = []
        for k in j:
            temp.append(np.array([k]))
        boxes.append(np.array(temp))

# Scale pixel value to 3d pos
scale = 100

# Height of waLL
wall_height = 1

# Создание вершин и граней для стен
verts, faces, wall_amount = transform.create_nx4_verts_and_faces(boxes, wall_height, scale)

# Создание вершин для верхних стен
verts = []
for box in boxes:
    verts.extend([transform.scale_point_to_vector(box, scale, 0)])

# Создание граней
faces = []
for room in verts:
    count = 0
    temp = ()
    for _ in room:
        temp = temp + (count,)
        count += 1
    faces.append([(temp)])


# Создаем директорию для сохранения изображений, если ее нет
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Генерация изображений на основе исходных данных
pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)

b = pol_room_seg.tolist() # nested lists with same data, indices
file_path = "output/test.json" ## your path variable
json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4) ### this saves the array in .json format