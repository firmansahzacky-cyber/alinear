import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Helper: Bentuk dasar
# =========================
def get_base_shape(shape_name: str) -> np.ndarray:
    """
    Mengembalikan koordinat bentuk dasar (Nx2).
    Default: persegi.
    """
    if shape_name == "Square":
        # Persegi dengan titik terakhir sama dengan titik pertama (biar tertutup)
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
        ], dtype=float)
    elif shape_name == "Triangle":
        points = np.array([
            [0, 0],
            [1, 0],
            [0.5, 1],
            [0, 0]
        ], dtype=float)
    else:
        # Default: persegi
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
        ], dtype=float)
    return points


# =========================
# Helper: Matriks Transformasi 3x3 (homogeneous)
# =========================
def translation_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ], dtype=float)


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=float)


def rotation_matrix(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)


def shearing_matrix(shx: float, shy: float) -> np.ndarray:
    # shearing di sumbu x dan y
    return np.array([
        [1,  shx, 0],
        [shy, 1,  0],
        [0,  0,   1]
    ], dtype=float)


def reflection_matrix(axis: str) -> np.ndarray:
    """
    Beberapa opsi refleksi:
    - x-axis
    - y-axis
    - origin
    - y = x
    - y = -x
    """
    if axis == "x-axis":
        mat = np.array([
            [1,  0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=float)
    elif axis == "y-axis":
        mat = np.array([
            [-1, 0, 0],
            [0,  1, 0],
            [0,  0, 1]
        ], dtype=float)
    elif axis == "origin":
        mat = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=float)
    elif axis == "y = x":
        mat = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)
    elif axis == "y = -x":
        mat = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0,  0, 1]
        ], dtype=float)
    else:
        mat = np.eye(3)
    return mat


# =========================
# Helper: Apply Transform
# =========================
def apply_transformation(points_2d: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    points_2d : array Nx2
    T         : matriks transformasi 3x3 (homogeneous)

    Mengembalikan titik hasil transformasi Nx2.
    """
    # tambahkan kolom ones untuk homogeneous coord
    ones = np.ones((points_2d.shape[0], 1), dtype=float)
    points_h = np.hstack([points_2d, ones])            # N x 3
    transformed_h = (T @ points_h.T).T                 # (3x3 @ 3xN)T -> N x 3

    # bagi dengan kolom terakhir (kalau bukan 1)
    w = transformed_h[:, 2:3]
    w[w == 0] = 1.0
    transformed_2d = transformed_h[:, :2] / w
    return transformed_2d


# =========================
# Main App
# =========================
def main():
    st.title("Matrix Transformation Web App")
    st.write(
        """
        Aplikasi ini menunjukkan **penerapan matriks transformasi 2D** 
        pada sebuah bentuk (square / triangle).  
        Fitur yang tersedia:
        1. Translation  
        2. Scaling  
        3. Rotation  
        4. Shearing  
        5. Reflection  
        """
    )

    # Sidebar: pilihan bentuk & transformasi
    st.sidebar.header("Pengaturan")
    shape_name = st.sidebar.selectbox(
        "Pilih bentuk awal",
        ["Square", "Triangle"]
    )

    transform_type = st.sidebar.selectbox(
        "Pilih jenis transformasi",
        ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"]
    )

    # Ambil titik bentuk dasar
    base_points = get_base_shape(shape_name)

    # Ambil parameter transformasi sesuai jenisnya
    T = np.eye(3)

    if transform_type == "Translation":
        st.subheader("Translation")
        dx = st.number_input("dx (geser sumbu x)", value=1.0, step=0.5)
        dy = st.number_input("dy (geser sumbu y)", value=1.0, step=0.5)
        T = translation_matrix(dx, dy)

    elif transform_type == "Scaling":
        st.subheader("Scaling")
        sx = st.number_input("sx (skala sumbu x)", value=1.5, step=0.1)
        sy = st.number_input("sy (skala sumbu y)", value=1.5, step=0.1)
        T = scaling_matrix(sx, sy)

    elif transform_type == "Rotation":
        st.subheader("Rotation")
        theta = st.number_input("Sudut rotasi (derajat)", value=45.0, step=5.0)
        T = rotation_matrix(theta)

    elif transform_type == "Shearing":
        st.subheader("Shearing")
        shx = st.number_input("Shear di sumbu x (shx)", value=0.5, step=0.1)
        shy = st.number_input("Shear di sumbu y (shy)", value=0.0, step=0.1)
        T = shearing_matrix(shx, shy)

    elif transform_type == "Reflection":
        st.subheader("Reflection")
        axis = st.selectbox(
            "Pilih sumbu / garis refleksi",
            ["x-axis", "y-axis", "origin", "y = x", "y = -x"]
        )
        T = reflection_matrix(axis)

    # Hitung hasil transformasi
    transformed_points = apply_transformation(base_points, T)

    # Tampilkan matriks transformasi
    st.markdown("### Matriks Transformasi (3x3)")
    st.write(T)

    # Plot sebelum & sesudah
    st.markdown("### Visualisasi Bentuk Sebelum dan Sesudah Transformasi")
    fig, ax = plt.subplots()
    # Bentuk awal
    ax.plot(base_points[:, 0], base_points[:, 1], marker="o", label="Original")
    # Bentuk hasil
    ax.plot(
        transformed_points[:, 0],
        transformed_points[:, 1],
        marker="o",
        linestyle="--",
        label="Transformed"
    )
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{transform_type} on {shape_name}")

    st.pyplot(fig)

    # Penjelasan singkat
    st.markdown("### Penjelasan Singkat")
    st.write(
        f"""
        - Bentuk awal yang digunakan: **{shape_name}**  
        - Jenis transformasi: **{transform_type}**  
        - Matriks transformasi di atas digunakan untuk mengalikan koordinat titik 
          dalam bentuk homogeneous (3x1).  
        - Hasil perkalian matriks memberikan koordinat baru (bentuk tertransformasi).
        """
    )


if __name__ == "__main__":
    main()
