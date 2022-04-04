import glob
import os
import os.path as Path
import urllib
from curses import meta

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detection.object_detection import detect_object

images: str = glob.glob(Path.join("data/test", "*.jpg"))
titles: dict = {
    1: u"person",
    2: u"bicycle",
    3: u"car",
    4: u"motorcycle",
    5: u"airplane",
    6: u"bus",
    7: u"train",
    8: u"truck",
    9: u"boat",
    10: u"traffic light",
    11: u"fire hydrant",
    12: u"stop sign",
    13: u"parking meter",
    14: u"bench",
    15: u"bird",
    16: u"cat",
    17: u"dog",
    18: u"horse",
    19: u"sheep",
    20: u"cow",
    21: u"elephant",
    22: u"bear",
    23: u"zebra",
    24: u"giraffe",
    25: u"backpack",
    26: u"umbrella",
    27: u"handbag",
    28: u"tie",
    29: u"suitcase",
    30: u"frisbee",
    31: u"skis",
    32: u"snowboard",
    33: u"sports ball",
    34: u"kite",
    35: u"baseball bat",
    36: u"baseball glove",
    37: u"skateboard",
    38: u"surfboard",
    39: u"tennis racket",
    40: u"bottle",
    41: u"wine glass",
    42: u"cup",
    43: u"fork",
    44: u"knife",
    45: u"spoon",
    46: u"bowl",
    47: u"banana",
    48: u"apple",
    49: u"sandwich",
    50: u"orange",
    51: u"broccoli",
    52: u"carrot",
    53: u"hot dog",
    54: u"pizza",
    55: u"donut",
    56: u"cake",
    57: u"chair",
    58: u"couch",
    59: u"potted plant",
    60: u"bed",
    61: u"dining table",
    62: u"toilet",
    63: u"tv",
    64: u"laptop",
    65: u"mouse",
    66: u"remote",
    67: u"keyboard",
    68: u"cell phone",
    69: u"microwave",
    70: u"oven",
    71: u"toaster",
    72: u"sink",
    73: u"refrigerator",
    74: u"book",
    75: u"clock",
    76: u"vase",
    77: u"scissors",
    78: u"teddy bear",
    79: u"hair drier",
    80: u"toothbrush",
}


def main():
    readme_text: str = st.markdown(open("README.md").read())
    st.sidebar.title("Меню")
    app_mode = st.sidebar.selectbox(
        "Выберите что сделать", ["О программе", "Запуск приложения", "Исходный код"]
    )
    if app_mode == "О программе":
        st.sidebar.success('Чтобы продолжить, выберите "Запуск приложения"')
    elif app_mode == "Исходный код":
        readme_text.empty()
        st.title("Код главного файла")
        st.code(open("myselfdrive.py").read())
        st.title("Код файла предсказаний")
        st.code(open("detection/object_detection.py").read())
    elif app_mode == "Запуск приложения":
        readme_text.empty()
        run_the_app()


def run_the_app():
    selected_frame_index, selected_frame = frame_selector_ui(DATA_ROOT)
    orig_img = load_image(selected_frame_index)
    orig_img = cv2.resize(orig_img, dsize=(704, 704), interpolation=cv2.INTER_AREA)
    image = st.image(orig_img)
    image.empty()
    precision = precision_value()
    object_selector(orig_img, precision)


def object_selector(img, precision: int):
    req_class = st.sidebar.selectbox("Выберите объект", list(titles.values()))
    image = detect_object(img, req_class, precision)
    return st.image(image)


def frame_selector_ui(data_root: str):
    selected_frame_index = st.sidebar.slider("Выберите картинку", 0, 256)
    selected_frame = images[selected_frame_index]
    return selected_frame_index, selected_frame


def precision_value():
    precision: float = st.sidebar.slider("Точность", 0.0, 1.0, 0.3)
    return precision


@st.cache(show_spinner=False)
def load_image(img_idx: int):
    image = images[img_idx]
    if image is not None:
        try:
            image = Image.open(image)
        except Exception:
            st.error("Ошибка, выбрана не картинка")
        else:
            img_array = np.array(image)
            return img_array
    return image


DATA_ROOT: str = "data/test"

if __name__ == "__main__":
    main()
