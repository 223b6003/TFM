import streamlit as st
import pandas as pd
import numpy as np
import joblib
import spacy
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
import streamlit_authenticator as stauth

# ---------------------------
# Configuración del layout
# ---------------------------
st.set_page_config(
    page_title="Agente de Vinculación PDET",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Autenticación de usuarios
# ---------------------------
# Credenciales de usuarios
credentials = {
    "usernames": {
        "admin": {
            "name": "Administrador",
            "password": "$2b$12$yJPCOjraPxLVx7qkL3qWM.kTFkvujc/0XPZ6TJTM7BVYuehngDphO"
        }
    }
}


authenticator = stauth.Authenticate(
    credentials, 'mi_app', 'auth_cookie', 1
)

name, authentication_status, username = authenticator.login('Iniciar sesión', 'main')

if authentication_status is False:
    st.error('Usuario o contraseña incorrectos')
    st.stop()
elif authentication_status is None:
    st.warning('Por favor ingrese sus credenciales')
    st.stop()
elif authentication_status:
    authenticator.logout('Cerrar sesión', 'sidebar')
    st.sidebar.success(f"Bienvenido, {name}")

# ---------------------------
# Cargar datos persistentes
# ---------------------------
@st.cache_data
def cargar_municipios():
    df = pd.read_csv("fuentes_informacion/MunicipiosColombia.csv", sep=";", encoding="utf-8", dtype=str)
    return df

@st.cache_resource
def cargar_modelo_y_vectores():
    modelo = Word2Vec.load("fuentes_informacion/modelo_iniciativas_w2v.model")
    df_vectores = joblib.load("fuentes_informacion/vectores_iniciativas.pkl")
    return modelo, df_vectores

df_municipios = cargar_municipios()
modelo_w2v, df_iniciativas = cargar_modelo_y_vectores()

@st.cache_resource
def cargar_pipeline():
    nlp = spacy.load("es_core_news_sm")
    bigram_model = joblib.load("fuentes_informacion/bigram_model.pkl")
    stopwords = joblib.load("fuentes_informacion/stopwords.pkl")
    return nlp, bigram_model, stopwords

nlp, bigram_model, stopwords = cargar_pipeline()

# ---------------------------
# Preprocesamiento
# ---------------------------
def preprocesar_texto(texto, nlp, bigram_model, stopwords):
    texto = unidecode(texto.lower())
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords]
    tokens = bigram_model[tokens]
    return tokens

def vector_promedio(tokens, modelo):
    vectores = [modelo.wv[token] for token in tokens if token in modelo.wv]
    return np.mean(vectores, axis=0) if vectores else np.zeros(modelo.vector_size)

def buscar_iniciativas_similares(texto, codigos_dane, modelo, df, top_n=10, umbral_similitud=0.85):
    tokens_input = preprocesar_texto(texto, nlp, bigram_model, stopwords)
    vector_input = vector_promedio(tokens_input, modelo)

    df["codigodane"] = df["codigodane"].astype(str).str.zfill(5)
    codigos_dane = [c.zfill(5) for c in codigos_dane]
    df_filtrado = df[df["codigodane"].isin(codigos_dane)].copy()

    st.subheader("Proyecto consultado:")
    st.write(f"**Nombre del Proyecto:** {nombre_proyecto}")
    st.write(f"**Descripción del Proyecto:** {descripcion_proyecto}")

    if df_filtrado.empty:
        st.warning("⚠️ No se encontraron iniciativas en los municipios seleccionados.")
        return df_filtrado
    else:
        st.info(f"Cantidad de iniciativas en municipios seleccionados: {len(df_filtrado)}")
        df_filtrado["similitud"] = df_filtrado["vector"].apply(lambda v: cosine_similarity([vector_input], [v])[0][0])
        df_filtrado = df_filtrado[df_filtrado["similitud"] >= umbral_similitud]
        df_ordenado = df_filtrado.sort_values(by="similitud", ascending=False).head(top_n)
        st.warning(f"Cantidad de iniciativas con similitud: {len(df_ordenado)}")
    return df_ordenado

# ---------------------------
# Interfaz de usuario
# ---------------------------
st.image("logo_ue.png", width=200)
st.title("Agente de vinculación de Proyectos de Inversión Pública a Iniciativas PDET")
st.write("Esta herramienta permite identificar las iniciativas comunitarias PDET más cercanas a un proyecto de inversión, según su descripción y ubicación.")
st.write("Por favor, ingrese los datos en el menú lateral para identificar las iniciativas asociadas.")

st.sidebar.header("Datos del Proyecto")
nombre_proyecto = st.sidebar.text_area("Nombre del Proyecto (*)")
descripcion_proyecto = st.sidebar.text_area("Descripción del Proyecto (*)")

departamentos_unicos = df_municipios[['DEPARTAMENTO', 'CODIGODANEDEPARTAMENTO']].drop_duplicates()
departamentos_seleccionados = st.sidebar.multiselect(
    "Seleccione uno o más departamentos",
    options=departamentos_unicos['DEPARTAMENTO'].tolist()
)

df_filtrado = df_municipios[df_municipios['DEPARTAMENTO'].isin(departamentos_seleccionados)].copy()
df_filtrado["municipio_mostrar"] = df_filtrado["MUNICIPIO"] + " (" + df_filtrado["CODIGODANEMUNICIPIO"] + ")"

municipios_seleccionados = st.sidebar.multiselect(
    "Seleccione uno o más municipios (*)",
    options=df_filtrado["municipio_mostrar"].tolist()
)

buscar = st.sidebar.button("🔍 Buscar iniciativas similares")

if "ver_detalle" not in st.session_state:
    st.session_state.ver_detalle = False

if buscar:
    campos_vacios = []

    if not nombre_proyecto.strip():
        campos_vacios.append("el nombre del proyecto")
    if not descripcion_proyecto.strip():
        campos_vacios.append("la descripción del proyecto")
    if not municipios_seleccionados:
        campos_vacios.append("al menos un municipio")

    if campos_vacios:
        mensaje = "⚠️ Por favor diligencie: " + ", ".join(campos_vacios) + "."
        st.warning(mensaje)
    else:
        st.success("✅ Procesando búsqueda de iniciativas similares...")
        codigos_dane = [s.split("(")[-1].replace(")", "") for s in municipios_seleccionados]
        texto_proyecto = nombre_proyecto + " " + descripcion_proyecto

        resultados = buscar_iniciativas_similares(
            texto_proyecto, codigos_dane, modelo_w2v, df_iniciativas, top_n=10, umbral_similitud=0.85
        )

        if resultados.empty:
            st.warning("⚠️ No se encontraron iniciativas similares con la información proporcionada.")
        else:
            mapeo_clusters = {
                0: "0 - Mejoramiento de infraestructura vial y conectividad territorial",
                1: "1 - Fortalecimiento del tejido social y comunitario con enfoque de género, paz y sostenibilidad",
                2: "2 - Fortalecimiento de sistemas productivos rurales y cadenas de valor",
                3: "3 - Garantía de acceso a servicios básicos de educación y salud"
            }
            resultados["cluster_fcm_c4"] = resultados["cluster_fcm_c4"].map(mapeo_clusters)

            columnas = [
                "código_iniciativa", 
                "subregión", 
                "municipio/sujeto_concertación", 
                "título_iniciativa", 
                "descripción_iniciativa", 
                "pilar", 
                "cluster_fcm_c4",
                "similitud"
            ]
            resultados = resultados[columnas]

            resultados = resultados.rename(columns={
                "código_iniciativa": "Código Iniciativa PDET",
                "subregión": "Subregión PDET",
                "municipio/sujeto_concertación": "Municipio / Sujeto de Concertación",
                "título_iniciativa": "Título de la Iniciativa PDET",
                "descripción_iniciativa": "Descripción de la Iniciativa PDET",
                "pilar": "Pilar PDET",
                "cluster_fcm_c4": "Cluster",
                "similitud": "Similitud"
            })

            resultados = resultados.sort_values(by="Similitud", ascending=False).reset_index(drop=True)

            st.subheader("Resultados detallados:")
            for idx, row in resultados.iterrows():
                with st.expander(f"📝 Código Iniciativa PDET: {row['Código Iniciativa PDET']} | Similitud: {row['Similitud']:.2%}"):
                    st.write(f"**Subregión PDET:** {row['Subregión PDET']}")
                    st.write(f"**Municipio / Sujeto de Concertación:** {row['Municipio / Sujeto de Concertación']}")
                    st.write(f"**Título de la Iniciativa PDET:** {row['Título de la Iniciativa PDET']}")
                    st.write(f"**Descripción de la Iniciativa PDET:** {row['Descripción de la Iniciativa PDET']}")
                    st.write(f"**Pilar PDET:** {row['Pilar PDET']}")
                    st.write(f"**Cluster:** {row['Cluster']}")
