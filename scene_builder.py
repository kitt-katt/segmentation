import json
import bpy
import numpy as np
from mathutils import Vector
from collections import deque
import bmesh


# Словарь классов
room_classes = {
    0.0: "Background",
    1.0: "Outdoor", 
    2.0: "Wall",  
    3.0: "Kitchen",  
    4.0: "Living Room",  
    5.0: "Bed Room",  
    6.0: "Bath",  
    7.0: "Entry",  
    8.0: "Railing",  
    9.0: "Storage",  
    10.0: "Garage", 
    11.0: "Undefined"  
}


def rename_and_filter_objects(obj, class_id):
    """
    Переименовывает объект на сцене Blender в соответствии с классом, исключая
    классы "Background" и "Outdoor".

    Parameters:
        obj (bpy.types.Object): объект в Blender, который нужно переименовать.
        class_id (float): идентификатор класса объекта.
        room_classes (dict): словарь классов с текстовыми названиями.
        
    Returns:
        bool: True если объект следует оставить, False если его нужно исключить.
    """
    # Исключаем классы "Background" и "Outdoor"
    if class_id in [0.0, 1.0]:  
        return False
    
    # Переименовываем объект по его классу
    class_name = room_classes.get(class_id, "Unknown")
    obj.name = f"{class_name}_Block"
    
    return True


# Загрузка JSON и создание numpy array
with open("output/test.json", "r") as file:
    data = json.load(file)
classification_array = np.array(data)



# Определение цветовой палитры для классов
class_colors = {
    2.0: (0, 0, 1, 1),    # Wall
    3.0: (1, 1, 0, 1),    # Kitchen
    4.0: (1, 0, 1, 1),    # Living Room
    5.0: (0, 1, 1, 1),    # Bed Room
    6.0: (0.5, 0.5, 0.5, 1),  # Bath
    7.0: (1, 0.5, 0, 1),  # Entry
    8.0: (0.5, 0, 0.5, 1),  # Railing
    9.0: (0, 0.5, 0.5, 1),  # Storage
    10.0: (0.5, 0.5, 0, 1), # Garage
    11.0: (1, 1, 1, 1),   # Undefined
}

# Функция для создания материала
def create_material(class_id):
    material_name = f"Class_{class_id}_Material"
    if material_name not in bpy.data.materials:
        mat = bpy.data.materials.new(name=material_name)
        mat.diffuse_color = class_colors[class_id]
        mat.use_nodes = False
    return bpy.data.materials[material_name]

# Функция для поиска блоков класса (поиск связанных компонент)
def find_blocks(array, class_id):
    visited = np.zeros(array.shape, dtype=bool)
    blocks = []
    
    def bfs(x, y):
        queue = deque([(x, y)])
        visited[x, y] = True
        block = [(x, y)]
        
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if 0 <= nx < array.shape[0] and 0 <= ny < array.shape[1]:
                    if not visited[nx, ny] and array[nx, ny] == class_id:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
                        block.append((nx, ny))
        return block
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == class_id and not visited[i, j]:
                block = bfs(i, j)
                blocks.append(block)
    return blocks

# Инициализация Blender сцены
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.delete(use_global=False)

# Создание мешей для каждого блока каждого класса
unique_classes = np.unique(classification_array)
for class_id in unique_classes:
    if class_id in [0.0, 1.0]:  # Пропускаем классы "Background" и "Outdoor"
        continue
    
    blocks = find_blocks(classification_array, class_id)
    
    for block in blocks:
        # Создаем новый меш для блока
        mesh = bpy.data.meshes.new(f"Class_{class_id}")
        obj = bpy.data.objects.new(f"Class_{class_id}", mesh)
        
        # Переименовываем объект и проверяем, нужно ли его оставить
        if not rename_and_filter_objects(obj, class_id):
            continue
        
        bpy.context.collection.objects.link(obj)
        
        # Создание bmesh для работы с произвольной геометрией блока
        bm = bmesh.new()
        
        # Добавление вершин в bmesh для каждой точки блока
        vert_map = {}
        for (y, x) in block:  # Invert coordinates for Blender
            vert = bm.verts.new((x, -y, 0))
            vert_map[(x, y)] = vert
        
        bm.verts.ensure_lookup_table()
        
        # Соединение вершин в грани
        for (x, y), vert in vert_map.items():
            face_verts = []
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if (nx, ny) in vert_map:
                    face_verts.append(vert_map[(nx, ny)])
            if len(face_verts) >= 3:  # Только если есть достаточно точек для грани
                bm.faces.new(face_verts)
        
        # Применение изменений к мешу
        bm.to_mesh(mesh)
        bm.free()
        
        # Назначение материала
        material = create_material(class_id)
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


# Сохранение файла Blender
save_path = "your_blender_file.blend"
bpy.ops.wm.save_as_mainfile(filepath=save_path)

print("Визуализация завершена и файл сохранен!")
