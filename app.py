import io
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# NLP
import nltk
import spacy
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("vader_lexicon")

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Audio Recording + Recognition
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment

# Visualization
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx
from pyvis.network import Network
from spacy import displacy


# ------------------------------------------------------------
# PAGE + BLUE THEME
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="NLPlayground")

st.markdown("""
<style>
body, .main, .block-container { background-color:#0d1117 !important; color:#e6edf3 !important; }

/* Blue Accent Buttons */
button[kind="primary"] {
  background:linear-gradient(90deg,#1f6feb,#58a6ff)!important; border:none!important;
  color:white!important; font-weight:600!important; box-shadow:0 0 18px #1f6feb88;
}
button[kind="primary"]:hover { box-shadow:0 0 30px #58a6ffcc; }

/* Card */
.card { background:#0b1220!important; border-radius:14px; padding:18px; border:1px solid #1f2a44; }

/* Scrollbar */
::-webkit-scrollbar { width:8px; }
::-webkit-scrollbar-thumb { background:#2b3757; border-radius:6px; }

/* NER tags */
.ent { padding:3px 6px; border-radius:6px; margin:0 2px; border:1px solid #1f2a44; background:rgba(88,166,255,0.2); color:#cde3ff; }

/* spaCy SVG Transparency */
svg { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>NLPlayground</h1>", unsafe_allow_html=True)
st.write("---")


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")


# ------------------------------------------------------------
# ANALYZER
# ------------------------------------------------------------
class NLPAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()

    def tokenize(self, text):
        return nltk.sent_tokenize(text), nltk.word_tokenize(text)

    def stemming(self, text):
        words = nltk.word_tokenize(text)
        return [(w, self.stemmer.stem(w)) for w in words if w.isalpha()]

    def lemmatization(self, text):
        doc = nlp(text)
        return [(t.text, t.lemma_) for t in doc if t.text.isalpha()]

    def pos(self, text):
        return nltk.pos_tag(nltk.word_tokenize(text))

    def ner(self, text):
        doc = nlp(text)
        return doc, [(e.text, e.label_) for e in doc.ents]

    def sentiment(self, text):
        s = self.sia.polarity_scores(text)
        label = "Positive" if s["compound"] >= 0.05 else "Negative" if s["compound"] <= -0.05 else "Neutral"
        return label, s

    def ngrams(self, text, n):
        words = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

    def build_graph(self, grams):
        G = nx.DiGraph()
        for g in grams:
            if len(g)==2: G.add_edge(g[0], g[1])
            if len(g)==3: G.add_edge(g[0], g[1]); G.add_edge(g[1], g[2])
        return G


analyzer = NLPAnalyzer()


# ------------------------------------------------------------
# INPUT (MIC OR TYPING)
# ------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Input Text")

audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", use_container_width=True)
text_prefill = ""

if audio:
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio["bytes"]), format="webm")
    except:
        seg = AudioSegment.from_file(io.BytesIO(audio["bytes"]), format="ogg")

    seg = seg.set_frame_rate(16000).set_channels(1)
    seg.export("temp.wav", format="wav")
    st.audio("temp.wav")

    r = sr.Recognizer()
    with sr.AudioFile("temp.wav") as src:
        rec = r.record(src)
    try:
        text_prefill = r.recognize_google(rec)
        st.success("Speech → Text completed.")
    except:
        st.error("Speech not recognized.")

text = st.text_area("Text to analyze:", value=text_prefill, height=150)
run = st.button("Run NLP Analysis")

st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# ANALYSIS PIPELINE
# ------------------------------------------------------------
if run and text.strip():
    
    sents, words = analyzer.tokenize(text)

    # Tokenize / Stemming / Lemma
    c1, c2, c3 = st.columns(3)

    with c1: st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Tokenization"); st.write(sents); st.write(words); st.markdown("</div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Stemming"); st.dataframe(analyzer.stemming(text), use_container_width=True, height=250); st.markdown("</div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='card'>", unsafe_allow_html=True); st.subheader("Lemmatization"); st.dataframe(analyzer.lemmatization(text), use_container_width=True, height=250); st.markdown("</div>", unsafe_allow_html=True)

    # POS + Sentiment
    st.write("")
    c4, c5 = st.columns([1.25, 1])

    with c4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Part-of-Speech")
        pos_df = pd.DataFrame(analyzer.pos(text), columns=["Word","POS"])
        st.dataframe(pos_df, use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    with c5:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Sentiment Analysis")
        label, s = analyzer.sentiment(text)
        st.write(f"Overall: **{label}**")
        sent_df = pd.DataFrame({"Sentiment":["Pos","Neu","Neg"], "Score":[s["pos"], s["neu"], s["neg"]]})
        st.plotly_chart(px.bar(sent_df, x="Sentiment", y="Score"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # NER
    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Named Entity Recognition")
    doc, ents = analyzer.ner(text)
    st.dataframe(pd.DataFrame(ents, columns=["Entity","Label"]), use_container_width=True, height=240)
    ner_html = displacy.render(doc, style="ent", options={"bg":"transparent"})
    components.html(f"<div style='padding:8px'>{ner_html}</div>", height=260, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # DEPENDENCY — (B, C, D) TABS
    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dependency Parsing Visualizations")
    tabs = st.tabs(["B) Syntax Tree", "C) Dependency Network Graph", "D) Constituency Tree"])

    # B) Hierarchical Tree
    with tabs[0]:
        G = nx.DiGraph()
        for token in doc:
            if token.dep_ != "ROOT":
                G.add_edge(token.head.text, token.text)
        fig = px.scatter()
        pos = nx.spring_layout(G, k=1.3, iterations=60)
        for n in G.nodes(): fig.add_scatter(x=[pos[n][0]], y=[pos[n][1]], text=[n], mode="text")
        for h,t in G.edges(): fig.add_shape(type="line", x0=pos[h][0], y0=pos[h][1], x1=pos[t][0], y1=pos[t][1])
        fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)

    # C) PyVis Dependency Net
    with tabs[1]:
        G = nx.DiGraph()
        for t in doc:
            G.add_node(t.text)
            if t.dep_ != "ROOT": G.add_edge(t.head.text, t.text)
        net = Network(height="500px", width="100%", directed=True)
        net.from_nx(G); net.repulsion(node_distance=180, spring_length=200)
        net.save_graph("dep_graph.html")
        components.html(open("dep_graph.html").read(), height=540, scrolling=True)

    # D) Constituency Tree
    with tabs[2]:
        from nltk import pos_tag, RegexpParser
        tagged = pos_tag(nltk.word_tokenize(text))
        grammar = r"NP: {<DT>?<JJ>*<NN.*>+}"
        tree = RegexpParser(grammar).parse(tagged)
        st.text(tree.pformat())

    st.markdown("</div>", unsafe_allow_html=True)

    # Word Cloud
    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Word Cloud")
    wc = WordCloud(width=900, height=420, background_color="black", colormap="Blues").generate(text)
    fig_wc = px.imshow(wc.to_array())
    fig_wc.update_xaxes(visible=False); fig_wc.update_yaxes(visible=False)
    st.plotly_chart(fig_wc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='text-align:center;'>made by KHUSH , DHRUV and Aryan</p>", unsafe_allow_html=True)
