import glob
import os
import os.path as Path
import urllib
from cProfile import run
from curses import meta

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detection.object_detection import detect_object

with open("yolo/coco.names") as file:
    lines = [line.rstrip() for line in file]
lines.insert(0, "anything")


def main():
    readme_text: str = st.markdown(open("README.md").read())
    st.sidebar.title("Меню")
    app_mode = st.sidebar.selectbox(
        "Выберите что сделать", ["О программе", "Запуск приложения", "Исходный код"]
    )
    if app_mode == "О программе":
        st.sidebar.success('Чтобы продолжить, выберите "Запуск приложения"')
        st.image("readme.png")
    elif app_mode == "Исходный код":
        readme_text.empty()
        st.title("Код главного файла")
        st.code(open("myselfdrive.py").read())
        st.title("Код файла предсказаний")
        st.code(open("detection/object_detection.py").read())
    elif app_mode == "Запуск приложения":
        readme_text.empty()
        image_pack = st.sidebar.selectbox(
            "Выберите набор картинок", ["Транспорты и пешеходы", "Случайные картинки"]
        )
        if image_pack == "Транспорты и пешеходы":
            DATA_ROOT: str = "data/test"
            run_the_app(DATA_ROOT)
        elif image_pack == "Случайные картинки":
            DATA_ROOT: str = "data/test2"
            run_the_app(DATA_ROOT)


def run_the_app(data_root: str):
    DATA_ROOT = data_root
    selected_frame_index, selected_frame = frame_selector_ui(DATA_ROOT)
    orig_img = load_image(selected_frame_index, DATA_ROOT)
    orig_img = cv2.resize(orig_img, dsize=(704, 704), interpolation=cv2.INTER_AREA)
    image = st.image(orig_img)
    image.empty()
    precision = precision_value()
    object_selector(orig_img, precision)


def object_selector(img, precision: int):
    req_class = st.sidebar.selectbox("Выберите объект", lines)
    image = detect_object(img, req_class, precision)
    return st.image(image)


def frame_selector_ui(data_root: str):
    images: str = glob.glob(Path.join(data_root, "*.jpg"))
    selected_frame_index = st.sidebar.slider("Выберите картинку", 0, 256)
    selected_frame = images[selected_frame_index]
    return selected_frame_index, selected_frame


def precision_value():
    precision: float = st.sidebar.slider("Точность", 0.0, 1.0, 0.3)
    return precision


@st.cache(show_spinner=False)
def load_image(img_idx: int, data_root: str):
    images: str = glob.glob(Path.join(data_root, "*.jpg"))
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


if __name__ == "__main__":
    main()
