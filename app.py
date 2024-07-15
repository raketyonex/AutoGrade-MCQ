import streamlit as st

import cv2
import numpy as np
from PIL import Image

from omr import OMR

def numeric(answer):
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return answer_map[answer]

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    
    st.sidebar.title("Set Jawaban Benar")
    ans1_15 = []
    ans16_30 = []
    for i in range(15):
        ans1_15.append(numeric(st.sidebar.selectbox(f"No. {i+1}", ['A', 'B', 'C', 'D'])))
    for i in range(15):
        ans16_30.append(numeric(st.sidebar.selectbox(f"No. {i+16}", ['A', 'B', 'C', 'D'])))
    
    # Video Capt
    img = st.camera_input("")
    if img is not None:
        st.image(img)

    # File Upload
    # img = st.file_uploader("upload")
    # if img is not None:
    #     st.image(img)

    if st.button('Nilai'):
        if img is not None:
            image = np.array(Image.open(img))
            imagegray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result_img, total_score, n1_15, n16_30 = OMR(imagegray, ans1_15, ans16_30)
    
            if result_img is not None:
                st.write("Nilai Siswa:")
                st.info(total_score)
                st.write("Koreksi Jawaban:")
                col1, col2 = st.columns(2)

                with col1:
                    for i in range(15):
                        if n1_15[i] == 1:
                            st.success(f"No. {i+1}: Benar")
                        else:
                            st.error(f"No. {i+1}: Salah")
                            
                with col2:
                    for i in range(15):
                        if n16_30[i] == 1:
                            st.success(f"Koreksi {i+16}: Benar")
                        else:
                            st.error(f"Koreksi {i+16}: Salah")

            else:
                st.write("Gambar Error, Tidak Bisa Mengkoreksi")

        else:
            st.error("Gagal memuat gambar. Silakan coba lagi.")

if __name__ == '__main__':
    main()