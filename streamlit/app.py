"""OralSense AI — professional Streamlit website with real inference & Grad-CAM.

All model inference uses frozen local artifacts from the user's Kaggle notebook.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import html

import pandas as pd
import streamlit as st

from inference import (
    CLASS_NAMES,
    OralSenseEngine,
    missing_model_files,
    open_image,
)

HERE = Path(__file__).resolve().parent
st.set_page_config(
    page_title="OralSense AI | Intelligent Oral Image Analysis",
    page_icon="🦷", layout="wide", initial_sidebar_state="expanded",
)
st.markdown(f"<style>{(HERE / 'assets' / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ConvNeXt, DINOv2 and the adaptive gate…")
def load_engine() -> OralSenseEngine:
    return OralSenseEngine()


def section(overline: str, title: str, note: str = "") -> None:
    st.markdown(
        '<div class="section-head"><div class="overline">' + html.escape(overline) +
        '</div><h2>' + html.escape(title) + '</h2><p>' + html.escape(note) +
        '</p></div>', unsafe_allow_html=True,
    )


def card(title: str, paragraph: str, marker: str = "✦") -> None:
    st.markdown(
        f'<div class="panel"><div class="step">{html.escape(marker)}</div>'
        f'<h3>{html.escape(title)}</h3><p>{html.escape(paragraph)}</p></div>',
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<div class="warning-note"><b>Educational research prototype.</b> '
        'This tool is not a medical device or diagnostic service. A model prediction, '
        'confidence score or heatmap cannot confirm or rule out disease. '
        'For health concerns, consult a qualified dental professional.</div>',
        unsafe_allow_html=True,
    )


def probability_bars(scores: list[float], *, muted: bool = False) -> None:
    ordered = sorted(zip(CLASS_NAMES, scores), key=lambda x: x[1], reverse=True)
    for name, score in ordered:
        pct = max(0, min(float(score) * 100, 100))
        cls = 'neutral' if muted else ''
        st.markdown(
            f'<div class="score-row"><div class="score-top">'
            f'<span>{html.escape(name)}</span><span>{pct:.2f}%</span></div>'
            f'<div class="score-track"><div class="score-fill {cls}" '
            f'style="width:{pct:.3f}%"></div></div></div>',
            unsafe_allow_html=True,
        )


def footer() -> None:
    st.markdown(
        '<div class="footer"><b>OralSense AI</b> · Uncertainty-aware dual-model classification research'
        '<br>Six oral-image categories · Internal test evaluation only · '
        'Not a substitute for dental assessment</div>', unsafe_allow_html=True,
    )


# -------------------- Sidebar / navigation --------------------
# Handle homepage navigation BEFORE the sidebar radio widget is instantiated.
if st.session_state.pop('navigate_to_analyzer', False):
    st.session_state['page_choice'] = 'Analyze Image'

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-icon">✦</span>'
        '<span class="brand-name">OralSense <span style="color:#2c9c78">AI</span></span>'
        '<div class="brand-under">Adaptive oral intelligence</div></div>',
        unsafe_allow_html=True,
    )
    chosen = st.radio(
        'Pages', ['Home', 'Analyze Image', 'Our Technology', 'Performance', 'Project Info'],
        label_visibility='collapsed', key='page_choice',
    )
    st.markdown('<div class="sidebar-bottom">◈ <b>Research model status</b></div>', unsafe_allow_html=True)
    missing = missing_model_files()
    if missing:
        st.markdown('<span class="status off">○ Models not installed</span>', unsafe_allow_html=True)
        st.caption('Add your Kaggle-trained backup to enable live predictions.')
    else:
        st.markdown('<span class="status on">● Models available</span>', unsafe_allow_html=True)
        st.caption('ConvNeXt · DINOv2–SVM · Adaptive gate')
    st.markdown(
        '<div class="sidebar-bottom">Designed for an academic mini project.<br>'
        'Your uploaded images are processed locally in memory; they are not saved by the app.</div>',
        unsafe_allow_html=True,
    )


# -------------------- Home --------------------
if chosen == 'Home':
    st.markdown('''
    <div class="hero"><div class="orb"></div>
      <div class="eyebrow">✦ AI-powered oral image research</div>
      <h1>Thoughtful intelligence<br>for a healthier tomorrow.</h1>
      <p>Meet OralSense AI — a research platform that combines two complementary
      vision models, uncertainty awareness and explainable AI to explore oral-image
      classification more responsibly.</p>
      <div class="tags"><span>✦ Adaptive AI</span><span>✦ Six oral classes</span>
      <span>✦ Visual explanations</span><span>✦ Expert referral</span></div>
    </div>''', unsafe_allow_html=True)
    st.markdown('<span class="kicker">A closer look at our research</span>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4, gap='small')
    a.metric('Images in dataset', '3,031')
    b.metric('Disease classes', '06')
    c.metric('Learned-gate test accuracy', '96.49%')
    d.metric('Adaptive test coverage', '83.51%')

    section('How it works', 'Intelligence in every step', 'One carefully designed workflow, from image upload to a responsible research outcome.')
    cols = st.columns(4, gap='small')
    with cols[0]: card('Upload image', 'Choose a clear oral photograph from your device.', '01')
    with cols[1]: card('Primary analysis', 'ConvNeXt-Tiny analyzes the image and estimates class scores.', '02')
    with cols[2]: card('Adaptive review', 'If uncertain, DINOv2–SVM and a learned gate assess the prediction.', '03')
    with cols[3]: card('Result & explanation', 'Receive a class prediction or expert-referral suggestion, plus Grad-CAM.', '04')

    section('Get started', 'Explore an image', 'Upload a photo to examine how our trained models behave.')
    if st.button('Open the image analyzer  →', type='primary'):
        st.session_state['navigate_to_analyzer'] = True
        st.rerun()
    st.markdown('<div class="highlights">' + ''.join(f'<span class="pill">{html.escape(n)}</span>' for n in CLASS_NAMES) + '</div>', unsafe_allow_html=True)
    disclaimer()
    footer()


# -------------------- Analyze --------------------
elif chosen == 'Analyze Image':
    st.markdown('<span class="kicker">Live inference · Explainable AI</span>', unsafe_allow_html=True)
    st.title('Oral image analyzer')
    st.markdown('<p class="page-lead">Upload a photograph, analyze it with our actual saved Kaggle models, and explore a Grad-CAM explanation of the primary model.</p>', unsafe_allow_html=True)

    if missing:
        st.warning('Live model inference is not active yet. Install the model artifacts from your ORALSENSE_FINAL_BACKUP.zip first.')
        with st.expander('How to connect the Kaggle-trained models', expanded=True):
            st.write('Copy your previously downloaded **ORALSENSE_FINAL_BACKUP.zip** to your computer (no need to upload it into Streamlit). From the website folder, run:')
            st.code('python setup_models.py "C:\\path\\to\\ORALSENSE_FINAL_BACKUP.zip"', language='bash')
            st.write('Then restart the website. Required files:')
            for label, file in missing.items():
                st.code(f'{label} → {file}', language='text')
        disclaimer()
        st.stop()

    input_col, upload_col = st.columns([1.05, 1], gap='large')
    with input_col:
        st.markdown('### Upload your image')
        uploaded = st.file_uploader(
            'JPG, PNG or WEBP · maximum 15 MB',
            type=['jpg', 'jpeg', 'png', 'webp'],
            help='Only upload images you have permission to use.',
        )
        st.markdown('<div class="info-note">For best results, use a clear, well-lit oral image. The application uses the same 224×224 neutral-padding and normalization as our Kaggle evaluation.</div>', unsafe_allow_html=True)
        if uploaded:
            image_bytes = uploaded.getvalue()
            digest = hashlib.sha256(image_bytes).hexdigest()
            try:
                uploaded_image = open_image(image_bytes)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        else:
            image_bytes = None
            digest = None
            uploaded_image = None

    with upload_col:
        st.markdown('### Image preview')
        if uploaded_image is not None:
            st.image(uploaded_image, use_container_width=True)
            st.caption(f'{uploaded_image.width} × {uploaded_image.height} pixels · image displayed without predicted annotations')
        else:
            st.markdown('<div class="panel" style="min-height:200px;display:flex;align-items:center;justify-content:center;color:#678773;text-align:center">✧ Your uploaded image will appear here.<br><br>Results are not generated until you click Analyze.</div>', unsafe_allow_html=True)
        analyze_clicked = st.button('✦ Analyze image', type='primary', disabled=image_bytes is None, use_container_width=True)

    if analyze_clicked and image_bytes is not None:
        st.session_state.pop('cam_data', None)
        st.session_state.pop('cam_digest', None)
        try:
            with st.spinner('Running the frozen ConvNeXt → DINOv2 → adaptive gate pipeline…'):
                result = load_engine().predict(image_bytes)
            st.session_state['prediction'] = result
            st.session_state['prediction_digest'] = digest
        except Exception as exc:
            st.error('Unable to run inference. Check the model ZIP, Python dependencies and terminal logs.')
            st.exception(exc)
            st.stop()

    if digest is None or st.session_state.get('prediction_digest') != digest:
        disclaimer()
        st.stop()

    prediction = st.session_state['prediction']
    st.divider()
    section('Your result', 'Adaptive analysis report', 'Output from the actual frozen model pipeline; not a clinical diagnosis.')
    if prediction['decision'] == 'REFER':
        st.markdown('''<div class="result ref"><div class="overline">◇ Caution · No automatic disease prediction accepted</div><div class="main">Professional review recommended</div><div class="description">The primary model flagged uncertainty, and the learned gate did not support accepting the specialist prediction. The system abstained rather than forcing a label.</div></div>''', unsafe_allow_html=True)
    elif prediction['decision'] == 'SPECIALIST':
        st.markdown(
            '<div class="result"><div class="overline">✦ Specialist prediction selected</div>'
            f'<div class="main">{html.escape(prediction["final_label"])}</div>'
            '<div class="description">The primary model was uncertain, so DINOv2–SVM was consulted and the learned gate selected its class label.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result"><div class="overline">✦ Primary prediction selected</div>'
            f'<div class="main">{html.escape(prediction["final_label"])}</div>'
            '<div class="description">ConvNeXt-Tiny passed the frozen primary confidence threshold, so no specialist inference was necessary.</div></div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric('ConvNeXt class', prediction['primary_label'])
    c2.metric('Primary model score', f'{prediction["primary_confidence"]:.2%}')
    c3.metric('Specialist used', 'Yes' if prediction['specialist_consulted'] else 'No')

    with st.expander('▸ Explore the adaptive decision', expanded=True):
        l, r = st.columns(2)
        l.write(f'**Primary confidence cutoff:** {prediction["confidence_threshold"]:.3f}')
        r.write(f'**Primary uncertain:** {"Yes" if prediction["uncertain"] else "No"}')
        l.write(f'**DINOv2–SVM prediction:** {prediction["specialist_label"] or "Not consulted"}')
        r.write(f'**Gate switched to specialist:** {"Yes" if prediction["switched"] else "No"}')
        if prediction['gate_score'] is not None:
            l.write(f'**Learned gate switch score:** {prediction["gate_score"]:.3f}')
            r.write(f'**Gate switching threshold:** {prediction["gate_threshold"]:.2f}')
        st.caption(f'Device: {prediction["device"].upper()} · Model runtime: {prediction["runtime_seconds"]:.2f}s · Scores are not clinically calibrated disease probabilities.')

    st.markdown('### Predicted class scores')
    left_scores, right_scores = st.columns(2, gap='large')
    with left_scores:
        st.markdown('<div class="tiny-title">Primary model · ConvNeXt-Tiny</div>', unsafe_allow_html=True)
        probability_bars(prediction['primary_probabilities'])
    with right_scores:
        st.markdown('<div class="tiny-title">Secondary model · DINOv2–SVM</div>', unsafe_allow_html=True)
        if prediction['specialist_probabilities'] is not None:
            probability_bars(prediction['specialist_probabilities'], muted=True)
        else:
            st.info('Specialist scores are unavailable because the primary model was confident. This is intentional conditional inference.')

    st.divider()
    section('Explainable AI', 'Grad-CAM: where ConvNeXt focused', 'See which regions influenced the primary model’s class score. Grad-CAM is generated from the real ConvNeXt gradients.')
    st.markdown('<div class="info-note">Grad-CAM explains <b>ConvNeXt-Tiny’s initial prediction</b>, even if the final result comes from DINOv2 or the system recommends referral. Colored regions are model attention cues, <b>not a lesion segmentation or a diagnosis.</b></div>', unsafe_allow_html=True)
    if st.button('◈ Generate Grad-CAM explanation', type='primary'):
        try:
            with st.spinner('Computing gradients and creating the attention heatmap…'):
                cam_data = load_engine().gradcam(image_bytes, prediction['primary_index'])
            st.session_state['cam_data'] = cam_data
            st.session_state['cam_digest'] = digest
        except Exception as exc:
            st.error('Grad-CAM could not be generated for this image.')
            st.exception(exc)

    if st.session_state.get('cam_digest') == digest and 'cam_data' in st.session_state:
        cam_data = st.session_state['cam_data']
        cols = st.columns(3, gap='medium')
        with cols[0]:
            st.image(cam_data.padded, caption='1 · Preprocessed oral image (224 × 224)', use_container_width=True)
        with cols[1]:
            st.image(cam_data.heatmap, caption='2 · Grad-CAM attention heatmap', use_container_width=True)
        with cols[2]:
            st.image(cam_data.overlay, caption='3 · Attention overlay on image', use_container_width=True)
        st.caption(f'Explained primary class: {cam_data.class_name}. Red/yellow suggests stronger positive influence on this class score; blue suggests less.')
        buffer = BytesIO()
        cam_data.overlay.save(buffer, format='PNG')
        st.download_button('↓ Save Grad-CAM overlay (PNG)', buffer.getvalue(),
                           file_name='OralSense_GradCAM_Overlay.png', mime='image/png')
    disclaimer()
    footer()


# -------------------- Technology --------------------
elif chosen == 'Our Technology':
    st.markdown('<span class="kicker">Behind the intelligence</span>', unsafe_allow_html=True)
    st.title('How OralSense AI works')
    st.markdown('<p class="page-lead">A dual-model research architecture built around model uncertainty, selective consultation and a learned switching policy.</p>', unsafe_allow_html=True)
    section('Model architecture', 'Designed to know when to refer')
    st.markdown('''<div class="panel" style="line-height:2.2;font-weight:650;color:#205a49;font-size:1.03rem;text-align:center;padding:1.4rem">
    Uploaded oral image <b>→</b> ConvNeXt-Tiny <b>→</b> Confidence check<br>
    <span style="color:#668877;font-size:.9rem">Confident: accept primary label</span><br>
    <span style="color:#176a53">Uncertain: run DINOv2–SVM → learned gate → specialist label or referral</span>
    </div>''', unsafe_allow_html=True)
    a,b,c = st.columns(3, gap='medium')
    with a: card('01 · ConvNeXt-Tiny', 'Our primary image classifier. It was selected after benchmarking six candidates on the common validation split.', 'A')
    with b: card('02 · DINOv2 + SVM', 'A frozen visual embedding model paired with a calibrated SVM, used only for uncertain images.', 'B')
    with c: card('03 · Learned gate', 'A logistic regression decision policy trained on group-aware out-of-fold predictions to assess switching.', 'C')
    section('Routing rules', 'Three possible outcomes')
    st.markdown('**Primary:** ConvNeXt confidence exceeds the frozen threshold (0.98296022).')
    st.markdown('**Specialist:** ConvNeXt is uncertain; DINOv2 makes a different prediction; and the gate switch score is at least 0.51.')
    st.markdown('**Refer:** ConvNeXt is uncertain, but the gate does not approve a specialist switch. No accepted disease label is shown.')
    section('Explainability', 'Grad-CAM is an explanation, not a diagnosis')
    st.write('Grad-CAM uses the gradients of the predicted primary class with respect to the final spatial feature stage in ConvNeXt. The heatmap highlights which input regions influenced the ConvNeXt class score. It cannot establish that a highlighted region is a lesion or explain the specialist/gate decision.')
    disclaimer()
    footer()


# -------------------- Performance --------------------
elif chosen == 'Performance':
    st.markdown('<span class="kicker">Evidence & evaluation</span>', unsafe_allow_html=True)
    st.title('Model performance')
    st.markdown('<p class="page-lead">Our frozen OralSense system was evaluated on a 485-image internal held-out test set. Comparisons use the same split.</p>', unsafe_allow_html=True)

    section('Held-out evaluation', 'Final test comparison', 'Accuracy for referral systems is measured on the accepted predictions only.')
    comparison = pd.DataFrame([
        {'System':'ConvNeXt-Tiny only', 'Accuracy':'95.46%', 'Coverage':'100%', 'Macro-F1':0.9029, 'Referrals':0},
        {'System':'ConvNeXt + learned gate', 'Accuracy':'96.49%', 'Coverage':'100%', 'Macro-F1':0.9238, 'Referrals':0},
        {'System':'ConvNeXt + referral', 'Accuracy':'99.49%', 'Coverage':'81.65%', 'Macro-F1':0.9791, 'Referrals':89},
        {'System':'Complete adaptive system', 'Accuracy':'99.01%', 'Coverage':'83.51%', 'Macro-F1':0.9685, 'Referrals':80},
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Primary correct', '463 / 485')
    c2.metric('Learned gate correct', '468 / 485')
    c3.metric('Specialist rescued', '7 cases')
    c4.metric('Specialist harmed', '2 cases')
    st.markdown('<div class="info-note">The learned gate improved full-coverage test accuracy by <b>5 images / 1.03 percentage points</b>. This is a small difference, not proof of statistically or clinically superior performance.</div>', unsafe_allow_html=True)

    section('Coverage & referral', 'How the full system handled 485 images')
    st.write('**396** primary predictions accepted · **9** specialist predictions accepted · **80** referred for expert review')
    st.progress(405 / 485, text='405 / 485 images accepted · 83.51% coverage')
    st.write('Of the 405 accepted predictions, **401 were correct** (99.01% accepted accuracy). The complete system did not label the remaining 80 images.')
    st.write('The simpler primary-only referral baseline achieved **99.49%** accepted accuracy at **81.65%** coverage. Our complete system accepted 9 more cases, but accepted accuracy was slightly lower.')

    section('Experimental model selection', 'Six models examined on validation')
    validation = pd.DataFrame([
        {'Model':'ConvNeXt-Tiny', 'Accuracy (%)':95.11, 'Macro-F1':0.9119},
        {'Model':'EfficientNet-B0', 'Accuracy (%)':94.00, 'Macro-F1':0.8940},
        {'Model':'DINOv2–SVM', 'Accuracy (%)':94.00, 'Macro-F1':0.8922},
        {'Model':'DenseNet121', 'Accuracy (%)':94.00, 'Macro-F1':0.8905},
        {'Model':'ResNet50', 'Accuracy (%)':93.56, 'Macro-F1':0.8892},
        {'Model':'MobileNetV3-Large', 'Accuracy (%)':92.44, 'Macro-F1':0.8823},
    ])
    st.dataframe(validation, hide_index=True, use_container_width=True)
    st.caption('Internal validation data: 450 images. Class imbalance, source-image artifacts and absence of patient identifiers limit generalization.')
    disclaimer()
    footer()


# -------------------- About --------------------
else:
    st.markdown('<span class="kicker">About the research</span>', unsafe_allow_html=True)
    st.title('About OralSense AI')
    st.markdown('<p class="page-lead">An academic mini-project exploring selective AI classification of oral photographs through primary–specialist model collaboration.</p>', unsafe_allow_html=True)
    section('Our goal', 'Useful predictions. Responsible uncertainty.')
    st.write('Oral conditions can share visual characteristics. A single image classifier can also be confidently incorrect. OralSense AI studies whether a second vision model, an adaptive gate, and the option to abstain can improve internal classification performance.')
    a, b, c = st.columns(3)
    with a: card('Dataset preparation', '13,862 raw files were investigated; a strict cleaned set of 3,031 images was retained across six categories.', '01')
    with b: card('Careful evaluation', 'Six approaches were compared on fixed group-disjoint train / validation / test partitions.', '02')
    with c: card('Research prototype', 'Models and thresholds were frozen before evaluation on the 485-image test set.', '03')
    section('Dataset categories', 'Six classes included')
    st.markdown('<div class="highlights">' + ''.join(f'<span class="pill">{html.escape(n)}</span>' for n in CLASS_NAMES) + '</div>', unsafe_allow_html=True)
    section('Limitations', 'What this research cannot establish')
    st.write('The original data has no verified patient identifiers, the image labels lack independent clinical confirmation, and the model has not been assessed with prospective clinical or external-center data. Uploaded images may be out-of-distribution. Confidence scores are not medical risk estimates, and Grad-CAM heatmaps are not lesion annotations.')
    disclaimer()
    footer()
