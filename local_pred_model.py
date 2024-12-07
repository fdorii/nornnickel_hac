import os
import sys
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO

def convert_mask_to_yolo(mask_image):
    """Конвертирует маску изображения в формат YOLO."""
    if mask_image is None:
        print("Не удалось обработать маску")
        return None

    black_mask = cv2.inRange(mask_image, (0, 0, 0), (50, 50, 50))
    new_image = np.ones_like(mask_image) * 255  # Начинаем с белого изображения
    new_image[black_mask > 0] = [0, 0, 0]  # Заменяем черные пиксели

    return new_image


def iou(y_true, y_pred, class_label):
    y_true = y_true == class_label
    y_pred = y_pred == class_label
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()

    return inter / (union + 1e-8)





def batch_iou(submit_dict, gt_path):
    """Сравнение предсказанных масок с эталонными и вычисление IoU."""
    res = []
    gt_masks_list = {fn for fn in os.listdir(gt_path) if fn.lower().endswith(".png") or fn.lower().endswith(".jpg")}
    for image_name, base64_mask_str in submit_dict.items():
        gt_mask_name = image_name.replace(".jpg", ".png")
        if gt_mask_name in gt_masks_list:
            sb_mask_bytes = base64.b64decode(base64_mask_str)
            sb_mask_img = cv2.imdecode(np.frombuffer(sb_mask_bytes, np.uint8), cv2.IMREAD_COLOR)

            gt_mask_path = os.path.join(gt_path, gt_mask_name)
            gt_mask_img = cv2.imread(gt_mask_path)

            gt_mask_yolo = convert_mask_to_yolo(gt_mask_img)
            sb_mask_yolo = convert_mask_to_yolo(sb_mask_img)

            if gt_mask_yolo is not None and sb_mask_yolo is not None:
                res.append((iou(gt_mask_yolo, sb_mask_yolo, 255) + iou(gt_mask_yolo, sb_mask_yolo, 0)) / 2)
    return res


def infer_image(model, image_path):
    """Выполняет инференс модели YOLO по одному изображению."""
    image = cv2.imread(image_path)
    return model(image)


def create_mask(image_path, results):
    """Создаёт маску на основе результатов предсказания модели YOLO."""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for result in results:
        masks = result.masks
        if masks is not None:
            for mask_array in masks.data:
                mask_i = mask_array.numpy()
                mask_i_resized = cv2.resize(mask_i, (width, height), interpolation=cv2.INTER_LINEAR)
                mask[mask_i_resized > 0] = 255

    return mask


def run_inference_and_evaluate(dataset_path, gt_path, model_path, output_path):
    """Основная функция, которая выполняет инференс и оценивает IoU."""
    # Загрузка модели
    model = YOLO(model_path)
    model.to('cpu')

    results_dict = {}

    for image_name in os.listdir(dataset_path):
        if image_name.lower().endswith(".jpg"):
            image_path = os.path.join(dataset_path, image_name)
            results = infer_image(model, image_path)
            mask = create_mask(image_path, results)

            _, encoded_img = cv2.imencode(".png", mask)
            encoded_str = base64.b64encode(encoded_img).decode('utf-8')
            results_dict[image_name] = encoded_str

    # Сохраняем результаты инференса в файл JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    # Вычисляем IoU на основе сохранённого JSON и GT-масок
    iou_scores = batch_iou(results_dict, gt_path)
    mean_iou = np.mean(iou_scores)
    print(f"Среднее значение IoU: {mean_iou:.4f}")
    return mean_iou


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Использование: python script.py <dataset_path> <gt_path> <model_path> <output_path>")
        # python test_soilitnet.py cv_open_dataset/open_img cv_open_dataset/open_msk runs/segment/train5/weights/best.pt submit.json
        sys.exit(1)

    dataset_path, gt_path, model_path, output_path = sys.argv[1:]
    run_inference_and_evaluate(dataset_path, gt_path, model_path, output_path)
