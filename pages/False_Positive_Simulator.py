import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# 1. Set Page Configuration
st.set_page_config(
    page_title="False Positive Simulator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Simulation Function (Vectorized for Speed) ---
@st.cache_data
def simulate_false_positive(p_A, threshold, simulations, samples, n):
    """
    Vectorized simulation to estimate the false positive rate.
    """
    alpha_prior, beta_prior = 1, 1
    rng = np.random.default_rng(seed=42)

    # Simulate all tests at once (no loop)
    # Both variants use the same true conversion rate, p_A
    conv_A = rng.binomial(n, p_A, size=simulations)
    conv_B = rng.binomial(n, p_A, size=simulations)

    # Calculate posteriors for all simulations
    alpha_post_A = alpha_prior + conv_A
    beta_post_A = beta_prior + n - conv_A
    alpha_post_B = alpha_prior + conv_B
    beta_post_B = beta_prior + n - conv_B

    # Draw samples from all posterior distributions
    samples_A = beta.rvs(alpha_post_A, beta_post_A, size=(samples, simulations), random_state=rng)
    samples_B = beta.rvs(alpha_post_B, beta_post_B, size=(samples, simulations), random_state=rng)

    # Calculate P(B > A) for each simulation
    prob_B_better = np.mean(samples_B > samples_A, axis=0)

    # The false positive rate is the proportion of simulations that incorrectly found a winner
    false_positive_rate = np.mean(prob_B_better > threshold)
    
    return false_positive_rate

# 2. Page Title and Introduction
st.title("🚨 False Positive Simulator")
st.markdown(
    "This tool helps you validate your decision rules by simulating the false positive rate of your test setup when there is **no real difference** between the variants."
)

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("Simulation Parameters")
    
    p_A = st.number_input(
        "Baseline conversion rate (p_A)", 
        min_value=0.001, max_value=0.99, value=0.05, step=0.001, format="%.3f", 
        help="The true conversion rate for both variants in this A/A test simulation."
    )
    thresh = st.slider(
        "Decision Threshold", 0.80, 0.99, 0.95, step=0.01,
        help="The P(B > A) threshold required to declare a winner. 95% is common."
    )
    n = st.number_input(
        "Sample Size per Variant", min_value=100, value=10000, step=100,
        help="The number of users that will be tested in each variant."
    )
    
    st.subheader("Simulation Quality")
    simulations = st.slider(
        "Number of A/A Tests to Simulate", 100, 5000, 1000, step=100,
        help="More simulations provide a more accurate estimate but are slower."
    )
    samples = st.slider(
        "Posterior Samples", 500, 5000, 1000, step=500,
        help="Samples drawn from each posterior. Default is usually sufficient."
    )

    st.markdown("---")
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    with st.spinner("Running A/A test simulations..."):
        fp_rate = simulate_false_positive(p_A, thresh, simulations, samples, n)

        st.subheader("Simulation Results")
        st.metric(
            label="Estimated False Positive Rate",
            value=f"{fp_rate:.2%}",
            help="The percentage of simulated A/A tests that incorrectly declared a winner based on your threshold."
        )

        if fp_rate > 0.05:
            st.warning("Your threshold may be too lenient, leading to more false positives than typically accepted (5%).")
        else:
            st.success("Your decision threshold appears to be well-calibrated, keeping false positives low.")

        st.subheader("Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["False Positive Rate"], [fp_rate], color='salmon', width=0.4, label="Simulated Rate")
        ax.axhline(0.05, color='red', linestyle='--', label='Typical 5% Risk Level')
        ax.set_ylim(0, max(fp_rate * 2, 0.1))
        ax.set_ylabel("Rate")
        ax.set_title("Estimated False Positive Rate vs. 5% Threshold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to estimate the false positive rate.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ How to interpret these results"):
    st.markdown("""
    #### What is a False Positive?
    A **false positive** (or Type I error) occurs when you conclude that your variant (B) is better than the control (A), when in reality, there is no difference between them. This simulation estimates how often that would happen with your current settings.

    #### How does this simulation work?
    It runs hundreds or thousands of simulated A/A tests where both "variants" have the exact same true conversion rate. It then checks what percentage of those tests would have been declared a "winner" based on your decision threshold (e.g., `P(B > A) > 95%`).

    #### Why is this important?
    This tool helps you understand the risk associated with your decision-making process. A well-calibrated test should have a low false positive rate. The standard in the industry is to accept a rate of **5% or less**. If your estimated rate is higher, it suggests your threshold for declaring a winner might be too low (too lenient), and you risk launching changes that have no real positive effect.
    """)
