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
# LANGUAGE TEXTS
# ======================================================
def get_texts(lang: str):
    """Return UI texts for selected language."""
    if lang == "Indonesia":
        return {
            "profile_title": "Profil",
            "language_label": "Pilih Bahasa",
            "group_name": "Nama Kelompok",
            "group_section": "👥 Anggota Kelompok",
            "member_name": "Nama Anggota ",
            "member_sid": "SID Anggota ",
            "name_label": "Nama",
            "nim_label": "NIM",
            "feature_label": "Pilih Fitur:",
            "upload_hint": "Upload gambar di area utama (kanan).",
            "main_title": "📐 Aplikasi Matriks dengan Gambar",
            "developed_by": "Dikembangkan oleh",
            "upload_label": "Upload Gambar (PNG / JPG / JPEG)",
            "settings_title": "⚙️ Pengaturan",
            "preview_title": "🖼️ Pratinjau",
            "shift_x": "Geser X (tx)",
            "shift_y": "Geser Y (ty)",
            "scale_x": "Skala X (sx)",
            "scale_y": "Skala Y (sy)",
            "angle": "Sudut (derajat)",
            "shear_x": "Shear X (shx)",
            "shear_y": "Shear Y (shy)",
            "reflection_axis": "Sumbu Refleksi",
            "blur": "Blur (Gaussian Blur)",
            "sharpen": "Tingkat ketajaman",
            "auto_bg": "Hapus background otomatis (deteksi dari tepi gambar)",
            "bg_sensitivity": "Sensitivitas background",
            "bg_sensitivity_help": "Nilai lebih besar = penghapusan background lebih agresif.",
            "original_caption": "Original",
            "result_ip_caption": "Hasil - Image Processing",
            "result_feature_caption": "Hasil - ",
        }
    elif lang == "Mandarin":
        return {
            "profile_title": "个人资料",
            "language_label": "选择语言",
            "group_name": "小组名称",
            "group_section": "👥 小组成员",
            "member_name": "成员姓名 ",
            "member_sid": "学号(SID) ",
            "name_label": "姓名",
            "nim_label": "学号",
            "feature_label": "选择功能:",
            "upload_hint": "在右侧主区域上传图片。",
            "main_title": "📐 矩阵图像应用程序",
            "developed_by": "开发者",
            "upload_label": "上传图片 (PNG / JPG / JPEG)",
            "settings_title": "⚙️ 设置",
            "preview_title": "🖼️ 预览",
            "shift_x": "水平平移 X (tx)",
            "shift_y": "垂直平移 Y (ty)",
            "scale_x": "缩放 X (sx)",
            "scale_y": "缩放 Y (sy)",
            "angle": "旋转角度 (度)",
            "shear_x": "剪切 X (shx)",
            "shear_y": "剪切 Y (shy)",
            "reflection_axis": "对称轴",
            "blur": "模糊 (Gaussian Blur)",
            "sharpen": "锐化程度",
            "auto_bg": "自动去除背景 (从边缘检测)",
            "bg_sensitivity": "背景灵敏度",
            "bg_sensitivity_help": "数值越大，去除背景越强。",
            "original_caption": "原图",
            "result_ip_caption": "结果 - 图像处理",
            "result_feature_caption": "结果 - ",
        }
    else:  # English default
        return {
            "profile_title": "Profile",
            "language_label": "Select Language",
            "group_name": "Group Name",
            "group_section": "👥 Group Members",
            "member_name": "Member Name ",
            "member_sid": "Member SID ",
            "name_label": "Name",
            "nim_label": "SID",
            "feature_label": "Choose Feature:",
            "upload_hint": "Upload image on the main area (right).",
            "main_title": "📐 Matrix Application with Image",
            "developed_by": "Developed by",
            "upload_label": "Upload Image (PNG / JPG / JPEG)",
            "settings_title": "⚙️ Settings",
            "preview_title": "🖼️ Preview",
            "shift_x": "Shift X (tx)",
            "shift_y": "Shift Y (ty)",
            "scale_x": "Scale X (sx)",
            "scale_y": "Scale Y (sy)",
            "angle": "Angle (degrees)",
            "shear_x": "Shear X (shx)",
            "shear_y": "Shear Y (shy)",
            "reflection_axis": "Reflection Axis",
            "blur": "Blur (Gaussian Blur)",
            "sharpen": "Sharpen level",
            "auto_bg": "Auto remove background (detect from borders)",
            "bg_sensitivity": "Background sensitivity",
            "bg_sensitivity_help": "Higher value = more aggressive background removal.",
            "original_caption": "Original",
            "result_ip_caption": "Result - Image Processing",
            "result_feature_caption": "Result - ",
        }

# ======================================================
# SIDEBAR (LANGUAGE + PROFILE + GROUP + FEATURE)
# ======================================================
with st.sidebar:
    # language selector
    lang = st.selectbox("🌐 Language / Bahasa / 语言", ["English", "Indonesia", "Mandarin"])
    t = get_texts(lang)

    st.title(f"👤 {t['profile_title']}")

    # group info
    group_name = st.text_input(t["group_name"], "")

    st.markdown("---")
    st.markdown(t["group_section"])
    member1_name = st.text_input(t["member_name"] + "1", "Zacky Firmansah")
    member1_sid = st.text_input(t["member_sid"] + "1", "004202400095")
    member2_name = st.text_input(t["member_name"] + "2", "")
    member2_sid = st.text_input(t["member_sid"] + "2", "")
    member3_name = st.text_input(t["member_name"] + "3", "")
    member3_sid = st.text_input(t["member_sid"] + "3", "")

    st.markdown("---")
    name = st.text_input(t["name_label"], member1_name)
    nim = st.text_input(t["nim_label"], member1_sid)

    st.markdown("---")
    feature = st.selectbox(
        t["feature_label"],
        ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "Image Processing"]
    )

    st.markdown("---")
    st.caption(t["upload_hint"])

# ======================================================
# MAIN TITLE
# ======================================================
st.title(t["main_title"])

caption_text = f"{t['developed_by']}: **{name}** | {t['nim_label']}: {nim}"
if group_name:
    caption_text += f" | {t['group_name']}: {group_name}"
st.caption(caption_text)

# ======================================================
# IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    t["upload_label"],
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
    st.subheader(t["settings_title"])
    M = None  # transformation matrix

    if feature == "Translation":
        tx = st.slider(t["shift_x"], -300, 300, 0)
        ty = st.slider(t["shift_y"], -300, 300, 0)

        M = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0,  1]
        ])

    elif feature == "Scaling":
        sx = st.slider(t["scale_x"], 0.1, 3.0, 1.0, 0.1)
        sy = st.slider(t["scale_y"], 0.1, 3.0, 1.0, 0.1)

        M = np.array([
            [sx, 0,  0],
            [0,  sy, 0],
            [0,  0,  1]
        ])

    elif feature == "Rotation":
        angle = st.slider(t["angle"], -180, 180, 0)
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
        shx = st.slider(t["shear_x"], -1.0, 1.0, 0.0, 0.05)
        shy = st.slider(t["shear_y"], -1.0, 1.0, 0.0, 0.05)

        M = np.array([
            [1,   shx, 0],
            [shy, 1,   0],
            [0,   0,   1]
        ])

    elif feature == "Reflection":
        axis = st.selectbox(t["reflection_axis"], ["X", "Y", "XY"])

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
        blur = st.slider(t["blur"], 0, 10, 0)
        sharpen = st.slider(t["sharpen"], 0, 5, 0)
        remove_bg = st.checkbox(t["auto_bg"], value=False)
        bg_sensitivity = st.slider(
            t["bg_sensitivity"], 10, 100, 40,
            help=t["bg_sensitivity_help"]
        )

# ======================================================
# PREVIEW
# ======================================================
with col_preview:
    st.subheader(t["preview_title"])

    if feature == "Image Processing":
        processed = img.copy()

        if blur > 0:
            processed = processed.filter(ImageFilter.GaussianBlur(radius=blur))

        for _ in range(sharpen):
            processed = processed.filter(ImageFilter.SHARPEN)

        if remove_bg:
            processed = remove_background_auto(processed, threshold=bg_sensitivity)

        c1, c2 = st.columns(2)
        c1.image(img, caption=t["original_caption"], use_column_width=True)
        c2.image(processed, caption=t["result_ip_caption"], use_column_width=True)

    else:
        transformed = apply_affine(img, M)

        c1, c2 = st.columns(2)
        c1.image(img, caption=t["original_caption"], use_column_width=True)
        c2.image(
            transformed,
            caption=t["result_feature_caption"] + feature,
            use_column_width=True
        )

        st.markdown("---")
        show_matrix(M)
