import streamlit as st

# Set page configuration at the top
st.set_page_config(
    page_title="Bayesian Testing Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Bayesian Testing Toolkit")

st.markdown("""
Welcome to the **Bayesian Testing Toolkit**. This suite of apps is designed to help you plan, interpret, and validate your A/B tests with statistical rigour. 
Select a tool from the sidebar to get started.
""")

st.markdown("---")

# Create a 2x2 grid for the tool cards
row1_col1, row1_col2 = st.columns(2, gap="large")
row2_col1, row2_col2 = st.columns(2, gap="large")

# --- Card 1: Uplift Certainty Estimator ---
with row1_col1:
    st.subheader("1. 📈 Uplift Certainty Estimator")
    st.markdown(
        "Interpret completed A/B test results to make a confident ship/no-ship decision."
    )
    st.markdown(
        """
        - **Input:** Users & conversions for each variant.
        - **Output:** Probability B > A, uplift estimate, and credible intervals.
        """
    )
    st.info(
        "**Note:** Ensure you use **user-level data** for an accurate Sample Ratio Mismatch (SRM) check.",
        icon="🧠"
    )

# --- Card 2: Pre-Test Calculator ---
with row1_col2:
    st.subheader("2. ⚙️ Pre-Test Calculator")
    st.markdown(
        "Plan your A/B test by estimating the sample size needed to detect a specific uplift."
    )
    st.markdown(
        """
        - **Input:** Baseline rate, expected uplift, & power target.
        - **Output:** The minimum required sample size per variant.
        """
    )
    st.info(
        "**Use when:** You're planning a test and want to ensure it's sufficiently powered to find a result.",
        icon="🧠"
    )

# --- Card 3: False Positive Simulator ---
with row2_col1:
    st.subheader("3. 🚨 False Positive Simulator")
    st.markdown(
        "Validate your decision rules by simulating the false positive rate under an A/A scenario."
    )
    st.markdown(
        """
        - **Input:** Conversion rate, decision threshold, & sample sizes.
        - **Output:** The estimated false positive rate.
        """
    )
    st.info(
        "**Use when:** You want to understand the risk of declaring a winner when none exists.",
        icon="🧠"
    )

# --- Card 4: SRM Calculator (NEW) ---
with row2_col2:
    st.subheader("4. ⚖️ SRM Calculator")
    st.markdown(
        "Check for imbalances in your test's traffic allocation that could invalidate results."
    )
    st.markdown(
        """
        - **Input:** Observed users and expected split for each variant.
        - **Output:** A p-value and conclusion on whether an SRM is detected.
        """
    )
    st.info(
        "**Use when:** You need to validate data integrity before trusting your test results.",
        icon="🧠"
    )
