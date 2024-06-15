import cv2


def crop_and_resize_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이진화하여 흰색 배경과 검은색 선을 분리합니다.
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # 비어있는 좌표를 초기화합니다.
    x_min, y_min = 256, 256
    x_max, y_max = 0, 0

    # 이미지의 모든 픽셀을 탐색하여 검은색 선이 있는 좌표를 찾습니다.
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 255:  # 검은색 선이 있는 좌표
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y

    # 찾은 좌표를 사용하여 이미지를 자릅니다.
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    # 자른 이미지를 256x256 크기로 조정합니다.
    resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)

    output_path = image_path
    cv2.imwrite(output_path, resized_image)
