import uuid

import tensorflow as tf
import os
from PIL import Image
import requests
from image_preprocess import crop_and_resize_image

# 사과, 자전거, 가위, 나무
# categories = ["사과", "자전거", "가위", "나무"]
categories = ["사과", "자전거", "뇌", "모니터", "라이플", "가위", "뱀", "나무"]


def predict_image(url: str):
    # 모델 불러오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, 'saved_model610_10_8.keras')
    model = tf.keras.models.load_model(model_save_path)

    image = Image.open(requests.get(url, stream=True).raw)
    temp_image = "temp/" + uuid.uuid4().__str__() + ".webp"
    image.save(temp_image)

    # image 전처리 진행
    crop_and_resize_image(temp_image)

    category = ""
    percentage = 0
    prediction_dict = {}

    try:
        # 이미지 불러오기
        image = tf.keras.preprocessing.image.load_img(temp_image, target_size=(256, 256))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, axis=0)  # 배치 차원 추가

        # 예측
        predictions = model.predict(image)
        category = categories[tf.math.argmax(predictions[0])]
        percentage = round(float(predictions[0][tf.math.argmax(predictions[0])]), 4)
        print(predictions, category, percentage)

        for index, item in enumerate(predictions[0].tolist()):
            prediction_dict[categories[index]] = round(float(item), 4)
    except:
        pass # TODO. implement error handler
    finally:
        try:
            os.remove(temp_image) # remove image after prediction
        except:
            pass

    return category, percentage, prediction_dict
