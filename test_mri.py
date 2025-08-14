import streamlit as st

def show_mri():
    st.header("🧪 Тест МРТ")
    st.success("МРТ-страница работает!")

page = st.sidebar.selectbox("Раздел", ["Главная", "МРТ"])

if page == "Главная":
    st.title("Главная")
else:
    show_mri()