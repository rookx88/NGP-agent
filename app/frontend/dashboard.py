import streamlit as st
from PIL import Image

st.set_page_config(page_title="Synch Dashboard", layout="wide")

# Custom CSS for Synch theme (purple/pink, torn-paper, neon)
st.markdown('''
    <style>
    body {
        background: linear-gradient(135deg, #d726a3 0%, #7e3ff2 100%) !important;
        color: #fff;
    }
    .block-container {
        background: #d726a3;
        border-radius: 24px;
        box-shadow: 0 4px 32px #7e3ff244;
        padding: 2rem 2rem 2rem 2rem;
    }
    .synch-title {
        font-family: 'Pacifico', cursive;
        font-size: 3.5rem;
        color: #fff;
        text-shadow: 0 0 16px #fff, 0 0 32px #ffb3e6;
        margin-bottom: 0.5em;
    }
    .neon {
        color: #fff;
        text-shadow: 0 0 8px #fff, 0 0 24px #ffb3e6;
    }
    .torn {
        border-radius: 50% 50% 48% 52% / 60% 40% 60% 40%;
        border: 8px solid #fff;
        box-shadow: 0 0 32px #ffb3e6;
        margin-bottom: 1em;
    }
    .panel-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #fff;
        text-shadow: 0 0 8px #ffb3e6;
        margin-bottom: 0.5em;
    }
    </style>
''', unsafe_allow_html=True)

# Synch logo/hero (use placeholder for now)
st.markdown('<div class="synch-title">Synch</div>', unsafe_allow_html=True)

# Layout: 2x2 grid
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="panel-title">🔥 Top Viral Tweets (Real-Time)</div>', unsafe_allow_html=True)
    st.write("Table of tweets sorted by virality score (coming soon)")
    # TODO: Add table with tweet text, handle, timestamp, velocity, likes, followers

    st.markdown('<div class="panel-title">📡 Tracked Accounts (Leaderboard)</div>', unsafe_allow_html=True)
    st.write("List of monitored accounts (coming soon)")
    # TODO: Add leaderboard with engagement velocity, viral count, signal score

with col2:
    st.markdown('<div class="panel-title">🧠 Emerging Keywords</div>', unsafe_allow_html=True)
    st.write("Word cloud or list of keywords (coming soon)")
    # TODO: Add word cloud/list with frequency, velocity, category filter

    st.markdown('<div class="panel-title">🧪 System Activity Log</div>', unsafe_allow_html=True)
    st.write("System activity log (coming soon)")
    # TODO: Add log of flagged tweets, keyword/account changes, auto-adds 