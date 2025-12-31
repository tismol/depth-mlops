import os
import io
import re
import zipfile
import base64
import os.path as op

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")


def zip_has_files(zip_bytes):
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        names_list = [n for n in zf.namelist() if n and not n.endswith("/")]
        return len(names_list) > 0
    except zipfile.BadZipFile:
        return False


def num_key(s: str):
    s = str(s)
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


st.set_page_config(page_title="Depth Service", layout="wide")
st.title("Depth prediction service")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Отправить изображение")
    f = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], key="single")

    if f is not None:
        file_bytes = f.getvalue()
        try:
            pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            st.image(pil, caption=f.name, width=320)
        except Exception as e:
            st.error(f"Не удалось открыть изображение: {e}")

        if st.button("Загрузить изображение"):
            try:
                r = requests.post(
                    f"{API_URL}/queue",
                    files={"image": (f.name, file_bytes, f.type)},
                    timeout=60
                )
                r.raise_for_status()
                st.success("Изображение отправлено в обработку.")
            except requests.RequestException as e:
                st.error(f"Ошибка при отправке: {e}")

    st.subheader("Отправить ZIP с изображениями")
    z = st.file_uploader("", type=["zip"], key="zip")

    if z is not None and st.button("Загрузить ZIP"):
        try:
            r = requests.post(
                f"{API_URL}/queue_zip",
                files={"zip_file": (z.name, z.getvalue(), "application/zip")},
                timeout=180
            )
            r.raise_for_status()
            st.success("ZIP отправлен в обработку.")
        except requests.RequestException as e:
            st.error(f"Ошибка при отправке ZIP: {e}")

    st.subheader("Скачать все обработанные depth-карты")
    if st.button("Скачать ZIP"):
        try:
            resp = requests.get(f"{API_URL}/download_processed_zip", timeout=180)
            resp.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Не удалось получить ZIP: {e}")
        else:
            if not resp.content or not zip_has_files(resp.content):
                st.warning("Нет обработанных изображений.")
            else:
                b64 = base64.b64encode(resp.content).decode("utf-8")
                components.html(
                    f"""
                    <a id="dl_all" download="processed_depth.zip"
                       href="data:application/zip;base64,{b64}"></a>
                    <script>document.getElementById("dl_all").click();</script>
                    """,
                    height=0
                )

with col2:
    st.subheader("История и просмотр по имени файла")
    st.button("Обновить список", key="refresh_items")
    show_all = st.checkbox("Показать все записи", key="show_all_items")
    if show_all:
        params = {}
    else:
        limit = st.slider("Сколько записей показывать", 10, 500, 200, key="limit_items")
        params = {"limit": limit}

    try:
        stats = requests.get(f"{API_URL}/stats", timeout=10).json()
        total = int(stats.get("total"))
        done = int(stats.get("done"))
        remaining = int(stats.get("remaining"))
        failed = int(stats.get("failed"))
        processing = int(stats.get("processing"))
        queued = int(stats.get("queued"))
    except Exception:
        total = done = remaining = failed = processing = queued = 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Всего", total)
    m2.metric("Обработано", done)
    m3.metric("Осталось", remaining)
    m4.metric("Ошибки", failed)

    if total > 0:
        st.progress(done / total)
        st.caption(f"Очередь: {queued} · В обработке: {processing}")

    try:
        resp = requests.get(f"{API_URL}/items", params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except requests.RequestException as e:
        st.error(f"Не удалось получить историю: {e}")
        items = []

    df = pd.DataFrame(items)

    if df.empty:
        st.info("Пока нет запросов.")
    else:
        st.dataframe(df)
        names = df["name"].tolist()
        names = sorted(names, key=num_key)

        name = st.selectbox("Выберите изображение", names, key="name_select")
        base = op.splitext(op.basename(name))[0]

        img_bytes = None
        dep_bytes = None
        npy_bytes = None

        try:
            img = requests.get(f"{API_URL}/image", params={"name": name}, timeout=30)
            if img.status_code == 200 and img.content:
                img_bytes = img.content
        except requests.RequestException as e:
            st.error(f"Ошибка загрузки входного изображения: {e}")

        try:
            dep = requests.get(f"{API_URL}/depth_png", params={"name": name}, timeout=30)
            if dep.status_code == 200 and dep.content:
                dep_bytes = dep.content
        except requests.RequestException as e:
            st.error(f"Ошибка загрузки depth: {e}")

        try:
            npy = requests.get(f"{API_URL}/depth_npy", params={"name": name}, timeout=30)
            if npy.status_code == 200 and npy.content:
                npy_bytes = npy.content
        except requests.RequestException:
            st.error(f"Ошибка загрузки depth: {e}")

        left, right = st.columns(2)

        with left:
            st.markdown("**Вход**")
            if img_bytes:
                try:
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
                    st.image(pil, width=224)
                except Exception as e:
                    st.error(f"Не удалось отрисовать входное изображение: {e}")
            else:
                st.warning("Входное изображение не найдено.")

        with right:
            st.markdown("**Depth preview**")
            if dep_bytes:
                try:
                    dep_img = Image.open(io.BytesIO(dep_bytes)).convert("RGB").resize((224, 224))
                    st.image(dep_img, width=224)
                except Exception:
                    st.image(dep_bytes, width=224)
            else:
                st.warning("Результат ещё не готов.")

        b_left, b_png, b_npy = st.columns([1, 0.5, 0.5])
        with b_left:
            if img_bytes:
                st.download_button(
                    label="Скачать вход",
                    data=img_bytes,
                    file_name=f"{base}.jpg",
                    mime="image/jpeg",
                    key=f"download_orig_{base}",
                )
        with b_png:
            if dep_bytes:
                st.download_button(
                    label="Скачать depth.png",
                    data=dep_bytes,
                    file_name=f"{base}_depth.png",
                    mime="image/png",
                    key=f"download_png_{base}",
                )
        with b_npy:
            if npy_bytes:
                st.download_button(
                    label="Скачать depth.npy",
                    data=npy_bytes,
                    file_name=f"{base}_depth.npy",
                    mime="application/octet-stream",
                    key=f"download_npy_{base}",
                )
