import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Matrix Application with Image",
    layout="wide"
)

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def load_image(uploaded_file):
    """Load image and convert to RGBA."""
    return Image.open(uploaded_file).convert("RGBA")


def apply_affine(img, M):
    """
    Apply 3x3 affine matrix M to image.
    PIL needs (a, b, c, d, e, f) from the top 2 rows.
    """
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]

    return img.transform(
        img.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC
    )


def detect_border_color(pil_img):
    """
    Estimate background color from image borders.
    Works well for pictures with a solid or almost-solid background.
    """
    img = pil_img.convert("RGBA")
    data = np.array(img)
    h, w, _ = data.shape

    # collect pixels from all borders (top, bottom, left, right)
    top = data[0, :, :3]
    bottom = data[h - 1, :, :3]
    left = data[:, 0, :3]
    right = data[:, w - 1, :3]

    border_pixels = np.concatenate([top, bottom, left, right], axis=0)

    # mean color of border as background color
    bg_color = border_pixels.mean(axis=0)
    return bg_color  # (R, G, B)


def remove_background_auto(pil_img, threshold=40):
    """
    Remove background based on border color similarity.
    Pixels close to the detected border color become transparent.
    threshold: larger -> more aggressive background removal.
    """
    img = pil_img.convert("RGBA")
    data = np.array(img).astype(np.int16)  # prevent overflow

    # detect background color from borders
    bg_r, bg_g, bg_b = detect_border_color(pil_img)

    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]

    # Euclidean distance in RGB space
    dist_sq = (r - bg_r) ** 2 + (g - bg_g) ** 2 + (b - bg_b) ** 2
    thr_sq = threshold ** 2

    # pixels similar to background => transparent
    mask = dist_sq < thr_sq
    a[mask] = 0

    data[..., 3] = a
    new_img = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), mode="RGBA")
    return new_img


def show_matrix(M):
    """Show matrix M as table + LaTeX."""
    st.markdown("### Transformation Matrix (M)")
    st.write(M)

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
# SIDEBAR (PROFILE + FEATURE)
# ======================================================
with st.sidebar:
    st.title("👤 Profile")

    name = st.text_input("Name", "Zacky Firmansah")
    nim = st.text_input("NIM", "004202400095")

    st.markdown("---")
    feature = st.selectbox(
        "Choose Feature:",
        ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "Image Processing"]
    )

    st.markdown("---")
    st.caption("Upload image on the main area (right).")

# ======================================================
# MAIN TITLE
# ======================================================
st.title("📐 Matrix Application with Image")
st.caption(f"Developed by: **{name}** | NIM: {nim}")

# ======================================================
# IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "Upload Image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Please upload an image first.")
    st.stop()

img = load_image(uploaded_file)
width, height = img.size  # used for reflection & rotation

# Layout: left = controls, right = preview
col_control, col_preview = st.columns([1, 2])

# ======================================================
# CONTROLS
# ======================================================
with col_control:
    st.subheader("⚙️ Settings")
    M = None  # transformation matrix

    if feature == "Translation":
        tx = st.slider("Shift X (tx)", -300, 300, 0)
        ty = st.slider("Shift Y (ty)", -300, 300, 0)

        M = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0,  1]
        ])

    elif feature == "Scaling":
        sx = st.slider("Scale X (sx)", 0.1, 3.0, 1.0, 0.1)
        sy = st.slider("Scale Y (sy)", 0.1, 3.0, 1.0, 0.1)

        M = np.array([
            [sx, 0,  0],
            [0,  sy, 0],
            [0,  0,  1]
        ])

    elif feature == "Rotation":
        angle = st.slider("Angle (degrees)", -180, 180, 0)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # rotate around image center
        cx, cy = width / 2, height / 2

        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0,   1]])

        R = np.array([[cos_a, -sin_a, 0],
                      [sin_a,  cos_a, 0],
                      [0,      0,     1]])

        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0,  1]])

        M = T2 @ R @ T1

    elif feature == "Shearing":
        shx = st.slider("Shear X (shx)", -1.0, 1.0, 0.0, 0.05)
        shy = st.slider("Shear Y (shy)", -1.0, 1.0, 0.0, 0.05)

        M = np.array([
            [1,   shx, 0],
            [shy, 1,   0],
            [0,   0,   1]
        ])

    elif feature == "Reflection":
        axis = st.selectbox("Reflection Axis", ["X", "Y", "XY"])

        if axis == "X":
            # vertical flip: y' = -y + height
            M = np.array([
                [1,  0,      0],
                [0, -1,  height],
                [0,  0,      1]
            ])
        elif axis == "Y":
            # horizontal mirror: x' = -x + width
            M = np.array([
                [-1, 0,   width],
                [0,  1,       0],
                [0,  0,       1]
            ])
        else:  # "XY"
            M = np.array([
                [-1, 0,   width],
                [0, -1, height],
                [0,  0,      1]
            ])

    elif feature == "Image Processing":
        blur = st.slider("Blur (Gaussian Blur)", 0, 10, 0)
        sharpen = st.slider("Sharpen level", 0, 5, 0)
        remove_bg = st.checkbox("Auto remove background (detect from borders)", value=False)
        bg_sensitivity = st.slider(
            "Background sensitivity", 10, 100, 40,
            help="Higher value = more aggressive background removal."
        )

# ======================================================
# PREVIEW
# ======================================================
with col_preview:
    st.subheader("🖼️ Preview")

    if feature == "Image Processing":
        processed = img.copy()

        if blur > 0:
            processed = processed.filter(ImageFilter.GaussianBlur(radius=blur))

        for _ in range(sharpen):
            processed = processed.filter(ImageFilter.SHARPEN)

        if remove_bg:
            processed = remove_background_auto(processed, threshold=bg_sensitivity)

        c1, c2 = st.columns(2)
        c1.image(img, caption="Original", use_column_width=True)
        c2.image(processed, caption="Result - Image Processing", use_column_width=True)

    else:
        transformed = apply_affine(img, M)

        c1, c2 = st.columns(2)
        c1.image(img, caption="Original", use_column_width=True)
        c2.image(transformed, caption=f"Result - {feature}", use_column_width=True)

        st.markdown("---")
        show_matrix(M)
