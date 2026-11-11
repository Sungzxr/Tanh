# tanh_app.py
# Streamlit app 3 of 3: Hyperbolic Tangent (Tanh) activation function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tanh Activation Function", layout="wide")
st.title("Hyperbolic Tangent (Tanh) Activation Function")
st.write(
    "Tanh is similar to sigmoid but maps inputs to the range (-1, 1). "
    "Because it is zero-centred, it usually trains better than sigmoid in hidden layers."
)
st.latex(r"\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}")


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2


# --- Sidebar controls ---
with st.sidebar:
    st.header("Neuron settings")
    x = st.slider("Input x", -5.0, 5.0, 1.5, 0.1)
    w = st.slider("Weight w", -3.0, 3.0, 1.0, 0.1)
    b = st.slider("Bias b", -3.0, 3.0, 0.0, 0.1)

    st.header("Plot settings")
    z_min, z_max = st.slider("z range", -10.0, 10.0, (-6.0, 6.0), 0.5)
    show_grad = st.checkbox("Show derivative", value=True)
    show_sigmoid = st.checkbox("Compare with sigmoid", value=False)

# --- One neuron: z = w*x + b, output = tanh(z) ---
z = w * x + b
out = float(tanh(z))
grad = float(tanh_grad(z))

st.subheader("1. One neuron calculation")
c1, c2, c3 = st.columns(3)
c1.metric("z = w·x + b", f"{z:.3f}")
c2.metric("tanh(z)", f"{out:.4f}")
c3.metric("Derivative at z", f"{grad:.4f}")
st.write(f"z = ({w:.1f} × {x:.1f}) + {b:.1f} = **{z:.3f}**  →  tanh(z) = **{out:.4f}**")

# --- Plot ---
st.subheader("2. Activation curve")
zs = np.linspace(z_min, z_max, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(zs, tanh(zs), label="tanh(z)", linewidth=2)
if show_grad:
    ax.plot(zs, tanh_grad(zs), "--", label="Derivative", linewidth=1.5)
if show_sigmoid:
    ax.plot(zs, 1 / (1 + np.exp(-zs)), ":", label="Sigmoid (for comparison)", linewidth=1.5)
ax.scatter([z], [out], color="red", zorder=5, label="Current neuron")
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("z (weighted sum)")
ax.set_ylabel("Output")
ax.set_title("Tanh activation function")
ax.set_ylim(-1.1, 1.1)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Value table ---
st.subheader("3. Sample values")
sample = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
st.dataframe(
    pd.DataFrame(
        {"z": sample, "tanh(z)": tanh(sample).round(4), "Derivative": tanh_grad(sample).round(4)}
    ),
    hide_index=True,
)

# --- Notes ---
st.subheader("4. Key properties")
st.markdown(
    """
- **Output range:** -1 to 1 (zero-centred)
- **Advantages:** smooth and differentiable, zero-centred output, stronger gradients than sigmoid (maximum 1.0 at z = 0).
- **Disadvantages:** still saturates for large |z|, so gradients can vanish in deep networks.
- **Relation to sigmoid:** tanh(z) = 2σ(2z) − 1.
- **Typical use:** hidden layers, and recurrent networks (RNN and LSTM).
"""
)
