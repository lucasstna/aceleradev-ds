import streamlit as st
import pandas as pd
def main(): 
    IRIS = pd.read_csv('/home/lucas/Documents/reps/codenation/Data-Science-Online/Semana 2/IRIS.csv')

    st.title('AceleraDev Data Science')

    st.header('Segunda semana')

    uploaded_file = st.file_uploader('Choose a file to upload', type= 'csv')

    if uploaded_file is not None:
        # lê o arquivo escolhido
        df = pd.read_csv(uploaded_file)

        # cria um slider pra modificar o número de exemplos mostrado
        slider = st.slider('Number of displayed samples', 1, df.shape[0])

        # mostra o arquivo lido
        st.dataframe(df.head(slider))

if __name__ == '__main__':
    main()