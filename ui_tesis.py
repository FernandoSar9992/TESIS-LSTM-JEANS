"""Presentación académica V9: componentes visuales, sin lógica de pronóstico."""
from html import escape
import streamlit as st


def install_theme():
    st.markdown("""
<style>
:root {
  --tesis-ink:#17283F; --tesis-navy:#10314B; --tesis-teal:#007A83;
  --tesis-lime:#D8F3E8; --tesis-muted:#637388; --tesis-line:#DEE7EC;
  --tesis-paper:#F7FAFC; --tesis-gold:#CF9650;
}
html, body, [class*="css"] { font-family:"Aptos","Segoe UI",Arial,sans-serif; }
.stApp { background:linear-gradient(180deg,#F6FAFD 0%,#FFFFFF 43%,#F8FBFD 100%); }
.main .block-container { max-width:1210px; padding-top:2.05rem; padding-bottom:5rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#F0F7F8 0%,#F9FBFE 100%); border-right:1px solid #DEE7ED; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#183A52; }
h1,h2,h3,h4 { font-family:"Aptos Display","Segoe UI",Arial,sans-serif; letter-spacing:-.028em; color:var(--tesis-ink); }
h2 { font-weight:740; } h3 { font-weight:700; }
p,div,li { line-height:1.52; }
[data-testid="stMetric"] { border:1px solid var(--tesis-line); border-radius:16px; background:#fff; padding:15px 18px; box-shadow:0 6px 21px rgba(17,53,77,.035); }
[data-testid="stMetricLabel"] { color:var(--tesis-muted);font-size:.85rem; }
[data-testid="stMetricValue"] { color:var(--tesis-navy);font-weight:760;letter-spacing:-.06em; }
[data-testid="stFileUploader"] {border:1px solid var(--tesis-line);border-radius:14px;padding:6px;background:rgba(255,255,255,.6);}
div[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:17px !important;}
div[data-testid="stDataFrame"] {border:1px solid var(--tesis-line);border-radius:13px;overflow:hidden;}
button[data-testid="stBaseButton-primary"],button[kind="primary"] {
   background:linear-gradient(120deg,#007D85,#0C6678)!important;
   border:0!important;color:#FFFFFF!important;
   border-radius:12px!important; font-weight:750!important;
   box-shadow:0 8px 24px rgba(0,111,120,.18)!important;
}
button[data-testid="stBaseButton-primary"]:hover,button[kind="primary"]:hover{
   background:linear-gradient(120deg,#006C73,#074E61)!important;
   color:white!important;
}
button[data-testid="stBaseButton-secondary"] {border-radius:11px!important;border-color:#C9D8E1!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  gap:.35rem;border-bottom:1px solid #DDE8ED;
  background:#F2F7F9;padding:.38rem .5rem;border-radius:12px 12px 0 0;
}
[data-testid="stTabs"] button[role="tab"]{
  border-radius:9px!important;color:#516C78!important;
  font-weight:700!important;padding:.62rem .9rem!important;
}
[data-testid="stTabs"] button[aria-selected="true"]{
  background:white!important;color:#0A6874!important;
  box-shadow:0 2px 8px rgba(18,71,83,.07);
}
[data-testid="stAlert"] {border-radius:12px!important;}
[data-testid="stExpander"] details {border:1px solid #DCE7EC;border-radius:12px;background:#fff;}
[data-testid="stMarkdownContainer"] a{color:#006D82;}
.tesis-hero {
  position:relative; isolation:isolate; overflow:hidden; border-radius:21px;
  padding:32px 38px 30px;margin-bottom:24px;
  background:linear-gradient(118deg,#122A45 0%,#145269 74%,#166B74 100%);
  box-shadow:0 17px 42px rgba(10,49,71,.13);
}
.tesis-hero:before,.tesis-hero:after {
  content:"";position:absolute;z-index:-1;border-radius:100%;pointer-events:none;
  border:1px solid rgba(198,240,231,.17);
}
.tesis-hero:before{width:365px;height:365px;right:-140px;top:-225px;box-shadow:0 0 0 66px rgba(146,221,218,.055),0 0 0 135px rgba(146,221,218,.025);}
.tesis-hero:after{width:190px;height:190px;right:150px;bottom:-155px;box-shadow:0 0 0 48px rgba(146,221,218,.04);}
.tesis-overline{
  font-size:.72rem;font-weight:800;letter-spacing:.18em;
  text-transform:uppercase;color:#A9E4DD;margin-bottom:12px;
}
.tesis-hero h1{color:white!important;font-size:clamp(1.75rem,3vw,2.9rem);
  line-height:1.10;letter-spacing:-.045em;margin:0 0 13px;max-width:890px;}
.tesis-hero p{color:#D5E9EF;font-size:1.01rem;max-width:830px;margin:0 0 20px;}
.tesis-pills{display:flex;gap:9px;flex-wrap:wrap;}
.tesis-pill{font-size:.77rem;border:1px solid rgba(199,233,232,.24);
  padding:7px 11px;border-radius:999px;color:#F0FCFC;
  background:rgba(215,252,250,.065);font-weight:650;}
.tesis-note{border:1px solid #CBE4E6;border-left:5px solid #11818C;background:#F1FAFA;
  border-radius:12px;padding:15px 18px;color:#1C5660;margin:14px 0 20px;}
.tesis-note strong{color:#154653;}
.tesis-note.warm {border-color:#F0E1CE;border-left-color:#C58C44;background:#FFF9F0;color:#745328;}
.tesis-lead{font-size:.76rem;letter-spacing:.14em;text-transform:uppercase;
 color:#0B7680;font-weight:850;margin:10px 0 7px;}
.tesis-section-head h2{margin:.12rem 0 .55rem;color:#1C3550;}
.tesis-section-head p{color:#65798A;margin-top:0;max-width:820px;}
.tesis-mini-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;margin:12px 0 22px;}
.tesis-mini{border:1px solid #DDE8EE;border-radius:15px;
  background:white;padding:18px 18px 16px;box-shadow:0 7px 19px rgba(18,67,88,.04);}
.tesis-mini .k{color:#698093;text-transform:uppercase;font-size:.68rem;
  letter-spacing:.12em;font-weight:850;}
.tesis-mini .v{margin:4px 0 2px;font-weight:830;font-size:1.85rem;letter-spacing:-.07em;
  color:#123B57;line-height:1.12;}
.tesis-mini .h{font-size:.81rem;color:#698093;}
.tesis-quiet{font-size:.83rem;color:#6B7E8D;}
.tesis-ribbon{display:flex;align-items:flex-start;gap:13px;border:1px solid #DBEAEF;
 background:white;border-radius:14px;padding:13px 17px;margin:7px 0 18px;}
.tesis-ribbon .ico{background:#E2F6F0;border-radius:11px;padding:8px 10px;color:#087F75;font-weight:900;}
.tesis-ribbon .content{font-size:.9rem;color:#435F71;}
.tesis-ribbon .content strong{color:#193C55;}
@media(max-width:700px){
.tesis-hero{padding:25px 22px}.tesis-hero p{font-size:.89rem}
.tesis-mini-grid{grid-template-columns:1fr}.tesis-pill{font-size:.68rem}
}
</style>
""",unsafe_allow_html=True)


def hero_main():
    st.markdown("""
<section class="tesis-hero">
<div class="tesis-overline">Ingeniería industrial · Sistema de planificación productiva</div>
<h1>Pronóstico de demanda de jeans</h1>
<p>Modelación LSTM V9 para maquila y corte propio, con evaluación temporal,
pronóstico a 12 semanas e integración a la planificación de producción.</p>
<div class="tesis-pills">
<span class="tesis-pill">LSTM V9 · Contexto estacional</span>
<span class="tesis-pill">Maquila + Corte propio</span>
<span class="tesis-pill">Evaluación a 12 semanas</span>
<span class="tesis-pill">Auditoría académica</span>
</div>
</section>
""", unsafe_allow_html=True)


def hero_audit():
    st.markdown("""
<section class="tesis-hero">
<div class="tesis-overline">Análisis de generalización · Evidencia metodológica</div>
<h1>Validación cruzada temporal V9</h1>
<p>Diagnóstico independiente de sobreajuste con ventana expansiva,
reentrenamiento por partición, curvas de aprendizaje y comparación estacional.</p>
<div class="tesis-pills">
<span class="tesis-pill">Fold externo · 12 semanas</span>
<span class="tesis-pill">TRAIN → VALIDACIÓN → TEST reservado</span>
<span class="tesis-pill">Early Stopping + Dropout + L2</span>
</div>
</section>
""",unsafe_allow_html=True)


def note(title,body,warm=False):
    extra = " warm" if warm else ""
    st.markdown(
        f'<div class="tesis-note{extra}"><strong>{escape(title)}</strong>'
        f'<br/>{escape(body)}</div>',
        unsafe_allow_html=True,
    )


def kpis(items):
    """items: [(label,value,caption), ...], rendered in rows of three."""
    for offset in range(0,len(items),3):
        entries=items[offset:offset+3]
        cards=[]
        for label,value,caption in entries:
            cards.append(
                f'<article class="tesis-mini"><div class="k">{escape(str(label))}</div>'
                f'<div class="v">{escape(str(value))}</div>'
                f'<div class="h">{escape(str(caption))}</div></article>'
            )
        st.markdown('<div class="tesis-mini-grid">'+"".join(cards)+'</div>',unsafe_allow_html=True)


def ribbon(title,body):
    st.markdown(
        f'<div class="tesis-ribbon"><span class="ico">✓</span>'
        f'<div class="content"><strong>{escape(title)}</strong><br/>'
        f'{escape(body)}</div></div>',
        unsafe_allow_html=True,
    )
