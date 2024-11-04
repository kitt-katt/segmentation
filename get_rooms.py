import json

json_path = "test.json"  # Обновите путь к JSON
with open(json_path, 'r') as f:
    pixel_data = json.load(f)

# Цвета для каждого класса
class_colors = {
    "Background": [0.0, 0.0, 0.0],
    "Outdoor": [0.3, 0.6, 0.1],
    "Wall": [0.6, 0.6, 0.6],
    "Kitchen": [0.9, 0.5, 0.3],
    "Living Room": [0.8, 0.4, 0.2],
    "Bed Room": [0.5, 0.7, 0.3],
    "Bath": [0.6, 0.8, 0.7],
    "Entry": [0.9, 0.9, 0.1],
    "Railing": [0.2, 0.2, 0.5],
    "Storage": [0.7, 0.7, 0.2],
    "Garage": [0.3, 0.3, 0.3],
    "Undefined": [1.0, 0.0, 0.0]
}

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

height = len(pixel_data)
width = len(pixel_data[0]) if height > 0 else 0

# Функция для нахождения граничных точек и изменения направления
def find_boundary_and_corners(shape_pixels):
    shape_set = set(shape_pixels)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    boundary_points = []
    corners = []

    # Найти граничные точки
    for (x, y) in shape_pixels:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in shape_set:  # Если сосед не в фигуре
                boundary_points.append((x, y))
                break

    # Отслеживание поворотов по границе
    for i in range(len(boundary_points)):
        current = boundary_points[i]
        next_point = boundary_points[(i + 1) % len(boundary_points)]
        prev_point = boundary_points[i - 1]
        
        # Рассчитываем векторы и их изменение направления
        vec1 = (current[0] - prev_point[0], current[1] - prev_point[1])
        vec2 = (next_point[0] - current[0], next_point[1] - current[1])
        
        # Если направление меняется, добавляем в углы
        if vec1 != vec2:
            corners.append(current)

    return corners

# Функция для поиска фигуры
def find_shape(x, y, class_value, visited):
    stack = [(x, y)]
    shape = []
    
    while stack:
        cx, cy = stack.pop()
        
        if (cx, cy) in visited or pixel_data[cx][cy] != class_value:
            continue
        
        visited.add((cx, cy))
        shape.append((cx, cy))
        
        neighbors = [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
        for nx, ny in neighbors:
            if 0 <= nx < height and 0 <= ny < width and (nx, ny) not in visited:
                stack.append((nx, ny))
    
    return shape

# Основной алгоритм
visited = set()
output_data = {}

for x in range(height):
    for y in range(width):
        pixel_value = pixel_data[x][y]
        
        if pixel_value in room_classes.keys() and (x, y) not in visited:
            class_name = room_classes[pixel_value]
            shape_pixels = find_shape(x, y, pixel_value, visited)
            
            if shape_pixels:
                corners = find_boundary_and_corners(shape_pixels)
                shape_data = {
                    "coordinates": [{"x": px, "y": py} for px, py in corners],
                    "color": class_colors[class_name]
                }
                
                # Динамическое название фигуры "{имя класса} №фигуры"
                shape_name = f"{class_name} {len([key for key in output_data.keys() if class_name in key]) + 1}"
                output_data[shape_name] = shape_data

# Запись в JSON
with open("output_shapes.json", "w") as outfile:
    json.dump(output_data, outfile, indent=4)

print("Output saved to output_shapes.json")
