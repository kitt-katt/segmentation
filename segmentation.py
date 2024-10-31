from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from floortrans.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
discrete_cmap()
from floortrans.post_prosessing import split_prediction, get_polygons, split_validation
from mpl_toolkits.axes_grid1 import AxesGrid

def load_model_to_cpu(model_path='model_best_val_loss_var.pkl', model_name='hg_furukawa_original', in_channels=51, n_classes=44):
    model = get_model(model_name, in_channels)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Model loaded on CPU.")
    return model

def load_image_data(data_folder='input/', data_file='input.txt'):
    normal_set = FloorplanSVG(data_folder, data_file, format='txt', original_size=True)
    data_loader = DataLoader(normal_set, batch_size=1, num_workers=0)
    data_iter = iter(data_loader)
    val = next(data_iter)
    return val

def plot_source_image(np_img, output_path='output/source_image.png'):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(np_img)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved source image to {output_path}")

def plot_rooms_and_walls(label_np, room_classes, output_path='output/rooms_and_walls.png'):
    from IPython.display import Image
    from IPython.core.display import HTML

    print("зашли в функцию plot_rooms_and_walls")
    n_rooms = len(room_classes)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    print("чекпоинт 1")
    rseg = ax.imshow(label_np[0], cmap='rooms', vmin=0, vmax=n_rooms-0.1)
    print("чекпоинт 2")
    cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=10)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved rooms and walls to {output_path}")

def plot_icons(label_np, icon_classes, output_path='output/icons.png'):
    n_icons = len(icon_classes)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(label_np[1], cmap='icons', vmin=0, vmax=n_icons - 0.1)
    cbar = plt.colorbar(iseg, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(icon_classes, fontsize=10)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved icons to {output_path}")


def perform_segmentation(val, model, room_classes, icon_classes):
    """
    Выполняет сегментацию изображения с учетом поворотов, сохраняет результаты в файлы.
    
    :param val: Данные изображения, загруженные из DataLoader
    :param model: Загруженная модель для сегментации
    :param room_classes: Классы комнат для сегментации
    :param icon_classes: Классы иконок для сегментации
    """
    # Извлекаем данные из val
    image = val['image']
    label = val['label']
    junctions = val['heatmaps']
    
    # Преобразуем изображение для визуализации
    np_img = np.moveaxis(image[0].cpu().numpy(), 0, -1) / 2 + 0.5
    plot_source_image(np_img)
    
    # Загружаем метки классов в формате numpy
    label_np = label.data.numpy()[0]
    plot_rooms_and_walls(label_np, room_classes)
    plot_icons(label_np, icon_classes)
    
    # Устанавливаем параметры
    rot = RotateNTurns()
    n_classes = len(room_classes) + len(icon_classes) + 1
    height, width = label_np.shape[1:]
    img_size = (height, width)
    prediction = torch.zeros([4, 44, height, width])  # Используем n_classes = 44
    
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    with torch.no_grad():
        for i, (forward, back) in enumerate(rotations):
            # Выполняем поворот изображения и предсказание модели
            rot_image = rot(image, 'tensor', forward)
            pred = model(rot_image)

            # Возвращаем предсказание к исходной ориентации
            pred = rot(pred, 'tensor', back)
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            
            # Сохраняем результат предсказания в итоговый массив
            prediction[i] = pred[0]
            print(f"Итерация {i}, prediction.shape = {prediction.shape}, pred.shape = {pred.shape}")
    
    # Усредняем предсказания по всем поворотам
    prediction = torch.mean(prediction, 0, True)
    
    # Разделение на классы комнат и иконок
    rooms_pred = F.softmax(prediction[0, 21:33], 0).cpu().numpy().argmax(0)
    icons_pred = F.softmax(prediction[0, 33:], 0).cpu().numpy().argmax(0)

    # Сохранение результата сегментации комнат
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(rooms_pred, cmap='rooms', vmin=0, vmax=len(room_classes) - 0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(len(room_classes)) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=10)
    plt.savefig('output/segmented_rooms.png')
    plt.close()
    print("Saved segmented rooms to output/segmented_rooms.png")

    # Сохранение результата сегментации иконок
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(icons_pred, cmap='icons', vmin=0, vmax=len(icon_classes) - 0.1)
    cbar = plt.colorbar(iseg, ticks=np.arange(len(icon_classes)) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(icon_classes, fontsize=10)
    plt.savefig('output/segmented_icons.png')
    plt.close()
    print("Saved segmented icons to output/segmented_icons.png")
    
    return img_size, prediction


def post_processed_polygons(prediction, img_size, split, room_classes, icon_classes, output_folder='output/'):
    """
    Выполняет постобработку предсказаний нейронной сети, создавая полигональные изображения сегментации.

    :param prediction: Предсказания модели
    :param img_size: Размер изображения (высота, ширина)
    :param split: Список с указанием количества классов для тепловых карт, комнат и иконок
    :param room_classes: Список классов комнат
    :param icon_classes: Список классов иконок
    :param output_folder: Папка для сохранения результатов
    """

    height, width = img_size
    n_rooms = len(room_classes)
    n_icons = len(icon_classes)
    
    # Разделяем предсказания на тепловые карты, комнаты и иконки
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    
    # Получаем полигональные контуры для комнат и иконок
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])
    
    # Преобразуем полигоны в изображения
    pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)
    
    # Сохраняем изображение сегментации комнат с полигонами
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    rseg = ax.imshow(pol_room_seg, cmap='rooms', vmin=0, vmax=n_rooms - 0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(room_classes, fontsize=10)
    plt.tight_layout()
    room_output_path = f"{output_folder}/polygon_rooms.png"
    plt.savefig(room_output_path)
    plt.close()
    print(f"Saved polygon room segmentation to {room_output_path}")
    
    # Сохраняем изображение сегментации иконок с полигонами
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    iseg = ax.imshow(pol_icon_seg, cmap='icons', vmin=0, vmax=n_icons - 0.1)
    cbar = plt.colorbar(iseg, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels(icon_classes, fontsize=10)
    plt.tight_layout()
    icon_output_path = f"{output_folder}/polygon_icons.png"
    plt.savefig(icon_output_path)
    plt.close()
    print(f"Saved polygon icon segmentation to {icon_output_path}")



def main():
    room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
    icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

    model = load_model_to_cpu()
    val = load_image_data()
    img_size, prediction = perform_segmentation(val, model, room_classes, icon_classes)
    post_processed_polygons(prediction, img_size, [21, 12, 11], room_classes, icon_classes)

if __name__ == "__main__":
    main()
