import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Rekomendasi Film",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎬"
)

# --- CUSTOM CSS (STYLING) ---
st.markdown("""
<style>
    /* Import Font Google */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Card Styling - Neutral/Light */
    div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }

    /* Judul Utama */
    h1 {
        color: #E50914; /* Netflix Red */
        font-weight: 700;
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #E50914, #ff6b6b);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("movies_clean.csv")
        return df
    except FileNotFoundError:
        st.error("File 'movies_clean.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return pd.DataFrame()

@st.cache_resource
def build_model(df):
    if df.empty:
        return None, None
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["genres"].fillna(""))
    sim = cosine_similarity(matrix, matrix)
    return sim

# Init
df_movies = load_data()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("🎛️ Panel Kontrol")
    
    film_pilihan = None
    if not df_movies.empty:
        # Dropdown Film
        daftar_film = df_movies['title'].tolist()
        film_pilihan = st.selectbox(
            "🎬 Pilih Film Favoritmu:",
            daftar_film,
            index=0
        )
        
        # Slider Jumlah Rekomendasi
        jumlah_rekomendasi = st.slider(
            "🔢 Jumlah Rekomendasi:",
            min_value=3, 
            max_value=12, 
            value=6,
            step=3
        )
        
        # Toggle Mode Presentasi (Optional)
        mode_presentasi = st.toggle("🎤 Mode Presentasi", value=True)
        
        st.markdown("---")
        st.info("💡 **Tips:** Pilih film yang generenya kamu suka, sistem akan mencari kemiripannya.")
        st.caption("© 2026 Sistem Rekomendasi Film")

# --- MAIN DASHBOARD ---
if df_movies.empty:
    st.warning("Data belum dimuat.")
else:
    # Bangun Model
    sim_matrix = build_model(df_movies)

    # Header & Metrics
    st.title("🎬 Sistem Rekomendasi Film")
    st.markdown("##### *Content-Based Filtering Dashboard*")
    
    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Film Database", value=f"{len(df_movies):,}")
    with col2:
        # Hitung unique genres sederhana
        all_genres = df_movies['genres'].str.split('|').explode().unique()
        st.metric(label="Total Genre", value=len(all_genres))
    with col3:
        st.metric(label="Algoritma", value="TF-IDF + Cosine")

    st.markdown("---")

    # Tabs
    tab_rek, tab_insight, tab_info = st.tabs(["� **Rekomendasi**", "📊 **Insight Data**", "ℹ️ **Tentang Sistem**"])

    # TAB 1: REKOMENDASI
    with tab_rek:
        if film_pilihan and sim_matrix is not None:
            try:
                # Cari Index
                idx = df_movies[df_movies['title'] == film_pilihan].index[0]
                
                # Hitung Similarity
                sim_scores = list(enumerate(sim_matrix[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1 : jumlah_rekomendasi + 1] # Skip diri sendiri
                
                movie_indices = [i[0] for i in sim_scores]
                
                # Tampilkan Konteks
                genre_film_pilihan = df_movies.loc[idx, 'genres']
                st.success(f"Karena kamu menyukai **{film_pilihan}** ({genre_film_pilihan})", icon="✅")
                
                # Grid Layout Cards
                # Menggunakan st.container dan columns untuk layout 'card'
                cols = st.columns(3)
                for i, m_idx in enumerate(movie_indices):
                    movie_title = df_movies.loc[m_idx, 'title']
                    movie_genre = df_movies.loc[m_idx, 'genres']
                    score = sim_scores[i][1]
                    
                    with cols[i % 3]:
                        # Card Container
                        with st.container(): # Fixed: Using standard container
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;">
                                <h4 style="color: #333333; margin-bottom: 5px;">{movie_title}</h4>
                                <p style="color: #666666; font-size: 0.9em; margin-bottom: 10px;">🎭 {movie_genre}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(score, text=f"Tingkat Kecocokan: {int(score*100)}%")
                            if mode_presentasi:
                                st.caption(f"💡 *Direkomendasikan karena kemiripan konten sebesar {int(score*100)}%*")
                            st.write("") # Spacer

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    # TAB 2: INSIGHTS
    with tab_insight:
        st.subheader("📊 Analisis Genre Populer")
        st.markdown("Visualisasi distribusi genre dalam database film.")
        
        # Data Preparation for Chart
        genre_data = df_movies['genres'].str.split('|').explode().value_counts().reset_index()
        genre_data.columns = ['Genre', 'Jumlah Film']
        
        # Altair Bar Chart
        chart = alt.Chart(genre_data.head(15)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Genre', sort='-y', axis=alt.Axis(labelAngle=-45)),
            y='Jumlah Film',
            color=alt.Color('Jumlah Film', scale=alt.Scale(scheme='reds'), legend=None),
            tooltip=['Genre', 'Jumlah Film']
        ).properties(
            height=400
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.info("ℹ️ Grafik ini menampilkan 15 genre yang paling sering muncul dalam database.")

    # TAB 3: TENTANG
    with tab_info:
        col_info_1, col_info_2 = st.columns([2, 1])
        
        with col_info_1:
            st.subheader("🧠 Cara Kerja Sistem")
            st.markdown("""
            Sistem ini menggunakan metode **Content-Based Filtering**. Sederhananya:
            
            1.  **Analisa Teks**: Sistem membaca "tags" atau genre dari setiap film.
            2.  **TF-IDF Vectorization**: Mengubah kata-kata genre menjadi angka (vektor) agar bisa dihitung secara matematis.
            3.  **Cosine Similarity**: Sistem mengukur "sudut" kedekatan antar film. Semakin kecil sudutnya, semakin mirip film tersebut.
            """)
            
            st.subheader("🛠️ Teknologi")
            st.code("Python, Streamlit, Pandas, Scikit-Learn", language="python")
            
        with col_info_2:
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0;">
                <h3 style="color: #333;">Presentasi Kuliah</h3>
                <p style="color: #666;">Dibuat untuk demo implementasi Sistem Rekomendasi Sederhana.</p>
                <p style="font-size: 2em;">🎓</p>
            </div>
            """, unsafe_allow_html=True)
