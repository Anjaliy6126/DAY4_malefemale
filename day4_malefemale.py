```python
# -*- coding: utf-8 -*-
"""DAY4_malefemale.ipynb"""

import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Female vs Male Classifier",
    page_icon="🚹🚺",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#edf4ff,#ffffff,#eef7ff);
}

/* Main Container */
.main .block-container{
    max-width:900px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Hide Streamlit Menu & Footer */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Title */
h1{
    text-align:center;
    color:#0f172a;
    font-weight:800;
}

/* Subheadings */
h2,h3{
    color:#1e3a8a;
}

/* File Uploader */
[data-testid="stFileUploader"]{
    background:white;
    border:2px dashed #2563eb;
    border-radius:15px;
    padding:15px;
}

/* Uploaded Image */
img{
    border-radius:15px;
    border:3px solid #2563eb;
}

/* Success Message */
[data-testid="stAlert"]{
    border-radius:12px;
}

/* Cards */
.card{
    background:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 6px 20px rgba(0,0,0,0.12);
}

/* Repository Box */
.repo{
    background:#ffffff;
    border-left:6px solid #2563eb;
    padding:15px;
    border-radius:10px;
    margin-top:25px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

/* Footer */
.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("female_male.pkl")

IMG_SIZE = 64

# -------------------------
# Title
# -------------------------
st.title("🚹🚺 Female vs Male Image Classifier")

st.markdown("""
<div style='text-align:center;font-size:18px'>
Upload an image and let the Machine Learning model predict whether the person is <b>Male</b> or <b>Female</b>.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader(
    "📂 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=320
    )

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized)
    resized = resized.flatten()

    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    st.markdown("---")

    # Prediction
    if prediction == 0:
        st.success("## 🚺 Prediction : Female")
        st.balloons()
        st.toast("🎉 Prediction Completed : Female", icon="🚺")
    else:
        st.success("## 🚹 Prediction : Male")
        st.balloons()
        st.toast("🎉 Prediction Completed : Male", icon="🚹")

    st.progress(float(max(probability)))

    st.markdown("## 📊 Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
        <h3>🚺 Female</h3>
        <h1>{probability[0]*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        <h3>🚹 Male</h3>
        <h1>{probability[1]*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# GitHub Repository
# -------------------------
st.markdown("---")

st.markdown("""
<div class="repo">

### 💻 Project Source Code

Want to explore the implementation, training process, and project files?

👉 <b>GitHub Repository:</b><br>

<a href="https://github.com/Anjaliy6126/DAY4_malefemale" target="_blank">
https://github.com/Anjaliy6126/DAY4_malefemale
</a>

</div>
""", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
<hr>
Made with ❤️ using <b>Python</b>, <b>Scikit-Learn</b> and <b>Streamlit</b><br>
© 2026 Female vs Male Image Classifier
</div>
""", unsafe_allow_html=True)
```
