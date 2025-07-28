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
Select a tool below to get started.
""")

st.markdown("---")

# Create columns for the tool cards
col1, col2, col3 = st.columns(3)

# --- Card 1: Uplift Certainty Estimator ---
with col1:
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
    # Use a markdown link for internal navigation
    st.markdown("#### [Open the Estimator &rarr;](Uplift_Certainty_Estimator)")


# --- Card 2: Sample Size Calculator ---
with col2:
    st.subheader("2. ⚙️ Pre-Test Power Calculator")
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
    # Use a markdown link for internal navigation
    st.markdown("#### [Open the Calculator &rarr;](Sample_Size_Calculator)")


# --- Card 3: False Positive Simulator ---
with col3:
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
    # Use a markdown link for internal navigation
    st.markdown("#### [Open the Simulator &rarr;](False_Positive_Simulator)")
👉 Select a page from the **sidebar** to get started.

""")
