import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

# ======================================================
# CONFIG DASAR
# ======================================================
st.set_page_config(
    page_title="Matrix Application with Image",
    layout="wide"
)

# ======================================================
# FUNGSI BANTU
# ======================================================
def load_image(uploaded_file):
    """Load gambar dan ubah ke RGBA."""
    return Image.open(uploaded_file).convert("RGBA")


def apply_affine(img, M):
    """
    Terapkan transformasi affin ke gambar.
    M adalah matriks 3x3.
    PIL butuh parameter (a,b,c,d,e,f) dari 2x3 pertama.
    """
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    return img.transform(
        img.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC
    )


def remove_white_bg(pil_img, threshold=240):
    """Menghapus background putih (sederhana) menjadi transparan."""
    img = pil_img.convert("RGBA")
    data = np.array(img)
    r, g, b, a = data.T

    mask = (r > threshold) & (g > threshold) & (b > threshold)
    data[..., -1][mask.T] = 0  # alpha = 0 (transparan)

    return Image.fromarray(data)


def show_matrix(M):
    """Tampilkan matriks M dalam bentuk tabel dan LaTeX tanpa error."""
    st.markdown("### 🔢 Matriks Transformasi (M)")

    # Tampil sebagai tabel angka
    st.write(M)

    # Tampil sebagai LaTeX (pakai string biasa, backslash di-escape)
    latex_matrix = (
        "M = "
        "\\begin{bmatrix}"
        f"{M[0,0]:.2f} & {M[0,1]:.2f} & {M[0,2]:.2f} \\\\ "
        f"{M[1,0]:.2f} & {M[1,1]:.2f} & {M[1,2]:.2f} \\\\ "
        f"{M[2,0]:.2f} & {M[2,1]:.2f} & {M[2,2]:.2f}"
        "\\end{bmatrix}"
    )

    st.latex(latex_matrix)

# ======================================================
# SIDEBAR (PROFIL + PEMILIHAN FITUR)
# ======================================================
with st.sidebar:
    st.title("👤 Profile")

    nama = st.text_input("Nama", "Zacky Firmansah")
    nim = st.text_input("NIM", "")

    st.markdown("---")
    fitur = st.selectbox(
        "Pilih Fitur:",
        [
            "Translation",
            "Scaling",
            "Rotation",
            "Shearing",
            "Reflection",
            "Image Processing"
        ]
    )

    st.markdown("---")
    st.caption("Upload gambar di bagian utama (kanan).")

# ======================================================
# JUDUL HALAMAN
# ======================================================
st.title("📐 Matrix Application with Image")
st.caption(
    f"Developed by: **{nama}**" + (f" | NIM: {nim}" if nim else "")
)

# ======================================================
# UPLOAD GAMBAR
# ======================================================
uploaded_file = st.file_uploader(
    "Upload Gambar (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Silakan upload gambar terlebih dahulu.")
    st.stop()

img = load_image(uploaded_file)

# Layout 2 kolom: kiri = kontrol, kanan = hasil
col_control, col_result = st.columns([1, 2])

# ======================================================
# KONTROL & PERHITUNGAN MATRIKS
# ======================================================
with col_control:
    st.subheader("⚙️ Pengaturan")

    M = None  # matriks transformasi untuk fitur matriks

    if fitur == "Translation":
        tx = st.slider("Geser X (tx)", -300, 300, 0)
        ty = st.slider("Geser Y (ty)", -300, 300, 0)

        M = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

    elif fitur == "Scaling":
        sx = st.slider("Scale X (sx)", 0.1, 3.0, 1.0)
        sy = st.slider("Scale Y (sy)", 0.1, 3.0, 1.0)

        M = np.array([
            [sx, 0,  0],
            [0,  sy, 0],
            [0,  0,  1]
        ])

    elif fitur == "Rotation":
        angle = st.slider("Sudut Rotasi (derajat)", -180, 180, 0)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        M = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])

    elif fitur == "Shearing":
        shx = st.slider("Shear X (shx)", -1.0, 1.0, 0.0, 0.05)
        shy = st.slider("Shear Y (shy)", -1.0, 1.0, 0.0, 0.05)

        M = np.array([
            [1,   shx, 0],
            [shy, 1,   0],
            [0,   0,   1]
        ])

    elif fitur == "Reflection":
        axis = st.selectbox("Pilih Sumbu Refleksi", ["X", "Y", "XY"])

        if axis == "X":
            M = np.array([
                [1,  0, 0],
                [0, -1, 0],
                [0,  0, 1]
            ])
        elif axis == "Y":
            M = np.array([
                [-1, 0, 0],
                [0,  1, 0],
                [0,  0, 1]
            ])
        else:  # XY
            M = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0,  0, 1]
            ])

    elif fitur == "Image Processing":
        blur = st.slider("Blur (Gaussian Blur)", 0, 10, 0)
        sharpen = st.slider("Sharpen (ketajaman)", 0, 5, 0)
        remove_bg = st.checkbox("Remove White Background", value=False)

# ======================================================
# HASIL / PREVIEW
# ======================================================
with col_result:
    st.subheader("🖼️ Preview")

    if fitur == "Image Processing":
        processed = img.copy()

        if blur > 0:
            processed = processed.filter(
                ImageFilter.GaussianBlur(radius=blur)
            )

        for _ in range(sharpen):
            processed = processed.filter(ImageFilter.SHARPEN)

        if remove_bg:
            processed = remove_white_bg(processed)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", use_column_width=True)
        col2.image(processed, caption="Processed", use_column_width=True)

    else:
        # Fitur matriks: gunakan M untuk transformasi
        transformed = apply_affine(img, M)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", use_column_width=True)
        col2.image(
            transformed,
            caption=f"Result - {fitur}",
            use_column_width=True
        )

        st.markdown("---")
        show_matrix(M)
