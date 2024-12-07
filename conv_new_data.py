from PIL import Image
import os

# Путь к папке с изображениями
input_folder = 'soiling_dataset/train/rgbLabels'
output_folder = './train/rgbLabels'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # Открываем изображение
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGB')

        # Преобразуем изображение в черно-белое: чёрный (0, 0, 0) остаётся чёрным, остальные цвета — белыми
        pixels = image.load()
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = pixels[x, y]
                if (r, g, b) == (0, 0, 0):  # Чёрный цвет (можно подстроить под другие оттенки)
                    pixels[x, y] = (0, 0, 0)  # Чёрный остаётся чёрным
                else:
                    pixels[x, y] = (255, 255, 255)  # Все остальные цвета становятся белыми

        # Сохраняем изменённое изображение
        output_path = os.path.join(output_folder, filename)
        image.save(output_path)

print("Обработка изображений завершена!")
