import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import streamlit as st

# 1. Ρύθμιση της σελίδας
st.set_page_config(
    page_title="Predator-Prey Simulation",
    page_icon="📈",
    layout="wide",
)

st.title("🔬 Προσομοίωση Πληθυσμών (Lotka-Volterra)")
st.write("Διαδραστικό εργαλείο επίλυσης μη γραμμικών διαφορικών εξισώσεων.")


# 2. Ορισμός του συστήματος διαφορικών εξισώσεων
# Χρησιμοποιούμε _t για να δηλώσουμε ότι ο χρόνος δεν χρησιμοποιείται άμεσα στον υπολογισμό
def lotka_volterra(_t, z, a, b, d, g):
    x, y = z
    dx_dt = a * x - b * x * y
    dy_dt = d * x * y - g * y
    return [dx_dt, dy_dt]


# 3. Sidebar: Παράμετροι Χρήστη
st.sidebar.header("⚙️ Παράμετροι Μοντέλου")

alpha = st.sidebar.slider("Ρυθμός αναπαραγωγής θηραμάτων (α)", 0.1, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("Ρυθμός θήρευσης (β)", 0.01, 0.5, 0.1, 0.01)
delta = st.sidebar.slider("Ρυθμός αναπαραγωγής θηρευτών (δ)", 0.01, 0.5, 0.075, 0.01)
gamma = st.sidebar.slider("Ρυθμός θνησιμότητας θηρευτών (γ)", 0.1, 2.0, 1.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("🌱 Αρχικές Συνθήκες")
x0 = st.sidebar.number_input("Αρχικός πληθυσμός θηραμάτων (x0)", value=10.0)
y0 = st.sidebar.number_input("Αρχικός πληθυσμός θηρευτών (y0)", value=5.0)
t_max = st.sidebar.slider("Χρονικό διάστημα (t)", 10, 100, 50)

# 4. Αριθμητική Επίλυση
t_eval = np.linspace(0, t_max, 1000)
sol = solve_ivp(
    lotka_volterra,
    [0, t_max],
    [x0, y0],
    args=(alpha, beta, delta, gamma),
    t_eval=t_eval,
)

# 5. Οπτικοποίηση Αποτελεσμάτων
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Εξέλιξη στο Χρόνο")
    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter(
            x=sol.t,
            y=sol.y[0],
            mode="lines",
            name="Θηράματα (x)",
            line=dict(color="#2ca02c", width=2),
        )
    )
    fig_time.add_trace(
        go.Scatter(
            x=sol.t,
            y=sol.y[1],
            mode="lines",
            name="Θηρευτές (y)",
            line=dict(color="#d62728", width=2),
        )
    )
    fig_time.update_layout(
        xaxis_title="Χρόνος (t)", yaxis_title="Πληθυσμός", template="plotly_white"
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.subheader("🔄 Διάγράμμα Φάσεων (Phase Space)")
    fig_phase = go.Figure()
    fig_phase.add_trace(
        go.Scatter(
            x=sol.y[0],
            y=sol.y[1],
            mode="lines",
            name="Τροχιά",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig_phase.update_layout(
        xaxis_title="Θηράματα (x)",
        yaxis_title="Θηρευτές (y)",
        template="plotly_white",
    )
    st.plotly_chart(fig_phase, use_container_width=True)