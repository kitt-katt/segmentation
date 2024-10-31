import torch
from floortrans.models import get_model

def load_model_to_cpu(model_path='model_best_val_loss_var.pkl', model_name='hg_furukawa_original', in_channels=51, n_classes=44):
    """
    Загружает предобученную модель на CPU с заданными параметрами.
    
    :param model_path: Путь к файлу модели (контрольной точке)
    :param model_name: Название модели для инициализации
    :param in_channels: Количество входных каналов модели
    :param n_classes: Количество выходных классов для сегментации
    :return: Инициализированная модель, загруженная на CPU и готовая к использованию
    """
    # Инициализация модели с заданным количеством каналов на входе
    model = get_model(model_name, in_channels)
    
    # Добавление сверточного слоя для сегментации с количеством классов n_classes
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    
    # Загрузка весов модели из контрольной точки на CPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    
    # Перевод модели в режим оценки
    model.eval()
    
    print("Model loaded on CPU.")
    return model

def main():
    # Загрузка модели
    model = load_model_to_cpu()

if __name__ == "__main__":
    main()
