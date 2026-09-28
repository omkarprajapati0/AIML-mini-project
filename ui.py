"""Presentation helpers for the FraudLens dashboard."""
from html import escape

import matplotlib.pyplot as plt
import streamlit as st

COLORS = ['#008b75', '#4e75cd', '#dd8652', '#9c6cbb', '#bd536d']


def configure_page():
    st.set_page_config(page_title='FraudLens · Transaction intelligence', page_icon='◈', layout='wide')
    st.markdown('''<style>
    .stApp {background:#f6f8fa;color:#202f40}
    .block-container {max-width:1480px;padding-top:2rem;padding-bottom:3rem}
    [data-testid="stSidebar"] {background:#edf2f5;border-right:1px solid #dce4ea}
    [data-testid="stSidebar"] .block-container {padding-top:1.5rem}
    h1,h2,h3 {letter-spacing:-.035em;color:#172a3b}
    h1 {font-weight:750!important} h3 {font-size:1.3rem!important}
    p,li {line-height:1.6}
    [data-testid="stMetric"] {background:white;border:1px solid #e0e7ed;border-radius:14px;padding:18px 20px;min-height:112px}
    [data-testid="stMetricLabel"] {color:#526479;font-size:.82rem}
    [data-testid="stMetricValue"] {color:#172f42;font-size:1.9rem;font-weight:650}
    .brand {font-size:1.6rem;font-weight:800;letter-spacing:-1px;color:#173749;margin:0}
    .brand span {color:#008b75}.brand-sub {color:#617588;font-size:.65rem;letter-spacing:1.9px;margin-top:2px}
    .hero {position:relative;overflow:hidden;background:#142b3b;border-radius:20px;padding:32px 36px;margin:6px 0 24px;color:white}
    .hero:after {content:'◈';position:absolute;right:42px;top:-62px;font-size:285px;color:#67dec3;opacity:.12;line-height:1.2;pointer-events:none}
    .hero h1 {color:#f4f9fa!important;font-size:2.6rem;line-height:1.15;margin:9px 0 12px;max-width:730px}
    .hero p {color:#bdcfdb;max-width:680px;font-size:.97rem;margin-bottom:13px}
    .eyebrow {font-size:.66rem;letter-spacing:2px;text-transform:uppercase;color:#79d9c5;font-weight:700}
    .tag {display:inline-block;background:#234853;border:1px solid #3b6872;color:#a5e7d9;padding:4px 10px;border-radius:6px;font-size:.7rem;margin-right:7px}
    .section-label {font-size:.7rem;letter-spacing:1.6px;font-weight:700;color:#648094;text-transform:uppercase;margin-bottom:6px}
    .callout {background:#e9f5f0;border:1px solid #c9e5d8;border-left:4px solid #008b75;border-radius:12px;padding:18px 22px;margin:10px 0 18px}
    .callout strong {color:#164d42}.callout p {color:#526d65;font-size:.88rem;margin:5px 0 0}
    .empty {background:white;border:1px dashed #cbd8e1;border-radius:16px;padding:38px;text-align:center;margin:15px 0}
    .empty h3 {margin:0 0 10px}.empty p {color:#657b8d;margin:0 auto;max-width:560px}
    .stButton>button,.stDownloadButton>button {border-radius:8px;font-weight:600;min-height:40px}
    [data-testid="stDataFrame"] {border-radius:10px}
    [data-testid="stHorizontalBlock"] {gap:1.3rem}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px}
    [data-testid="stCaptionContainer"] {color:#63768a}
    @media(max-width:700px) {.hero {padding:24px}.hero h1 {font-size:2rem}.hero:after {right:-50px;opacity:.06}.block-container {padding-top:1rem}}
    </style>''', unsafe_allow_html=True)
    plt.rcParams.update({
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'axes.edgecolor': '#dde5eb',
        'axes.labelcolor': '#556b7d', 'text.color': '#294256', 'xtick.color': '#637a8c',
        'ytick.color': '#637a8c', 'grid.color': '#e6edf2', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': '#ffffff',
    })


def section(kicker, title, description=None):
    st.markdown(f'<div class="section-label">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if description:
        st.caption(description)


def empty_state(title, description):
    st.markdown(f'<div class="empty"><h3>{escape(title)}</h3><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def callout(title, description):
    st.markdown(f'<div class="callout"><strong>{escape(title)}</strong><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def show_figure(fig):
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)
