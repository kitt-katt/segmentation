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
