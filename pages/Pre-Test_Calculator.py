import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# 1. Set Page Configuration
st.set_page_config(
    page_title="Power Calculator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Simulation Functions ---
@st.cache_data
def run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh):
    """
    Runs a single set of simulations for a given sample size and conversion rates.
    Returns the calculated power.
    """
    n_A = n
    n_B = n
    
    rng = np.random.default_rng(seed=42)
    conversions_A = rng.binomial(n_A, p_A, size=simulations)
    conversions_B = rng.binomial(n_B, p_B, size=simulations)

    alpha_post_A = alpha_prior + conversions_A
    beta_post_A = beta_prior + n_A - conversions_A
    alpha_post_B = alpha_prior + conversions_B
    beta_post_B = beta_prior + n_B - conversions_B

    post_samples_A = beta.rvs(alpha_post_A, beta_post_A, size=(samples, simulations), random_state=rng)
    post_samples_B = beta.rvs(alpha_post_B, beta_post_B, size=(samples, simulations), random_state=rng)

    prob_B_better = np.mean(post_samples_B > post_samples_A, axis=0)
    power = np.mean(prob_B_better > thresh)
    return power

@st.cache_data
def simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior):
    """
    Simulates power across a range of sample sizes to find the minimum
    sample size required to achieve the desired power.
    """
    p_B = p_A * (1 + uplift)
    if p_B > 1.0:
        st.error(f"Error: Uplift of {uplift:.2%} on baseline {p_A:.2%} results in a conversion rate > 100%. Please lower the uplift or baseline.")
        return []

    results = []
    n = 100
    power = 0
    MAX_SAMPLE_SIZE = 5_000_000

    with st.spinner("Searching for required sample size... This may take a moment."):
        while power < desired_power and n < MAX_SAMPLE_SIZE:
            power = run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((n, power))
            if power >= desired_power:
                break
            if n < 1000:
                n += 100
            elif n < 20000:
                n = int(n * 1.5)
            else:
                n = int(n * 1.25)
    return results

@st.cache_data
def simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n):
    """
    Simulates power across a range of uplifts (MDEs) for a fixed sample size.
    """
    results = []
    uplifts = np.linspace(0.01, 0.50, 20)

    with st.spinner("Running simulations for MDE..."):
        for uplift in uplifts:
            p_B = p_A * (1 + uplift)
            if p_B > 1.0:
                continue
            power = run_simulation(fixed_n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((uplift, power))
            if power >= desired_power:
                break
    return results

# 2. Page Title and Introduction
st.title("⚙️ Pre-Test Power Calculator")
st.markdown(
    "This tool helps you plan an A/B test by estimating the sample size required or the minimum effect you can detect."
)

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("Test Parameters")

    mode = st.radio(
        "Planning Mode",
        ["Estimate Sample Size", "Estimate MDE (Minimum Detectable Effect)"],
        help="Choose whether to estimate required sample size for a given uplift, or the minimum uplift detectable for a fixed sample size."
    )

    p_A = st.number_input(
        "Baseline conversion rate (p_A)", min_value=0.0001, max_value=0.999, value=0.05, step=0.001,
        format="%.4f",
        help="Conversion rate for your control variant (A), e.g., 5% = 0.050"
    )
    thresh = st.slider(
        "Posterior threshold (e.g., 0.95)", 0.5, 0.99, 0.95, step=0.01,
        help="Confidence level to declare a winner — usually 0.95 or 0.99"
    )
    desired_power = st.slider(
        "Desired power", 0.5, 0.99, 0.8, step=0.01,
        help="Minimum acceptable power of detecting a real uplift"
    )
    simulations = st.slider(
        "Simulations", 100, 2000, 300, step=100,
        help="How many test simulations to run"
    )
    samples = st.slider(
        "Posterior samples", 500, 3000, 1000, step=100,
        help="How many samples to draw from each posterior distribution"
    )

    if mode == "Estimate Sample Size":
        uplift = st.number_input(
            "Expected uplift (e.g., 0.10 = +10%)", min_value=0.0001, max_value=0.999, value=0.10, step=0.01,
            format="%.4f",
            help="Relative improvement expected in variant B over A"
        )
    else:
        fixed_n = st.number_input(
            "Fixed sample size per variant", min_value=100, value=10000, step=100,
            help="Fixed sample size used to determine the minimum detectable uplift."
        )

    # --- UPDATED PRIOR BELIEFS SECTION ---
    st.subheader("Optional Prior Beliefs")
    use_auto_prior = st.checkbox(
        "Calculate priors from historical data",
        help="Check this to calculate priors based on past conversions and sample size."
    )
    if use_auto_prior:
        hist_conv = st.number_input(
            "Historical Conversions (Successes)",
            min_value=0,
            value=50,
            step=1,
            help="The raw number of conversions or successes from your historical data."
        )
        hist_n = st.number_input(
            "Historical Total Sample Size (Users)",
            min_value=1,
            value=1000,
            step=1,
            help="The total number of users or observations in your historical data."
        )
        if hist_conv > hist_n:
            st.error("Historical conversions cannot exceed the total sample size.")
            st.stop()
        
        # New calculation based on raw values
        alpha_prior = hist_conv
        beta_prior = hist_n - hist_conv
    else:
        alpha_prior = st.number_input(
            "Alpha (prior successes)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="Manually set your prior belief in successes."
        )
        beta_prior = st.number_input(
            "Beta (prior failures)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="Manually set your prior belief in failures."
        )

    st.markdown("---")
    run_button = st.button("Run Calculation", type="primary", use_container_width=True)


# 4. Main Page for Displaying Outputs
st.markdown("---")

results_available = False

if run_button:
    if mode == "Estimate Sample Size":
        results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior)
        if results:
            x_vals, y_vals = zip(*results)
            results_available = True
            st.subheader("📈 Sample Size Estimation")
            if y_vals[-1] >= desired_power:
                st.success(f"✅ Estimated minimum sample size per group: **{x_vals[-1]:,}** (achieved {y_vals[-1]:.1%} power).")
            else:
                st.warning("Could not reach desired power. The uplift may be too small or the power target too high for a practical test.")
    else: # Estimate MDE Mode
        results = simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n)
        if results:
            x_vals, y_vals = zip(*results)
            results_available = True
            st.subheader("📉 Minimum Detectable Effect (MDE)")
            if y_vals[-1] >= desired_power:
                st.success(f"✅ Minimum detectable relative uplift: **{x_vals[-1]:.2%}** (achieved {y_vals[-1]:.1%} power).")
            else:
                st.warning("Simulation could not reach target power with the given sample size.")

    if results_available:
        st.subheader("Visualizations")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vals, y_vals, marker='o', label='Estimated Power')
        ax.axhline(desired_power, color='red', linestyle='--', label='Target Power')
        if mode == "Estimate Sample Size":
            ax.set_xlabel("Sample Size per Group")
            if len(x_vals) > 1:
                ax.set_xscale('log')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        else:
            ax.set_xlabel("Relative Uplift (MDE)")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
        ax.set_ylabel("Estimated Power")
        ax.set_title("Power vs. " + ("Sample Size" if mode == "Estimate Sample Size" else "MDE"))
        ax.grid(True, which="both", ls="--", c='0.7')
        ax.legend()
        st.pyplot(fig)

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Calculation' to see the results.")


# --- Time-Based Planning ---
st.markdown("---")
st.header("⏱️ Time-Based Planning")
weekly_traffic = st.number_input(
    "Estimated total weekly traffic to the experiment",
    min_value=1,
    value=20000,
    step=100,
    help="Enter the total number of users you expect to enter the experiment each week (before the 50/50 split)."
)

if results_available:
    st.subheader("🗓️ Duration Estimate")
    users_per_week_per_variant = weekly_traffic / 2
    if users_per_week_per_variant > 0:
        if mode == "Estimate Sample Size":
            if 'y_vals' in locals() and y_vals[-1] >= desired_power:
                required_sample_size = x_vals[-1]
                estimated_weeks = required_sample_size / users_per_week_per_variant
                st.info(f"To reach **{required_sample_size:,} users per variant**, you'll need to run this test for approximately **{estimated_weeks:.1f} weeks**.")
            else:
                st.warning("Cannot estimate duration because the target power was not reached.")
        else:
            required_sample_size = fixed_n
            estimated_weeks = required_sample_size / users_per_week_per_variant
            st.info(f"To reach **{required_sample_size:,} users per variant**, it will take approximately **{estimated_weeks:.1f} weeks**.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ Learn about the concepts used in this calculator"):
    st.markdown("""
    #### What is Sample Size? 👥
    **Sample size** is the number of users in each group of your test. Think of it like the lens on a camera you're using to see which variant is better. A bigger sample size gives you a bigger, more powerful lens.

    This bigger lens makes your test more sensitive in two key ways:

    1.  **You can spot smaller improvements (Lower MDE 🔎)**
        A powerful camera lens (**more users**) can spot a tiny, faint star that a weaker lens would miss. Similarly, a larger sample size allows you to reliably detect a very **small uplift (a lower MDE)**.

    2.  **You're more certain about what you see (Higher Power 💪)**
        When you're trying to photograph a specific star, a bigger lens (**more users**) gives you a much better chance (**higher power**) of capturing a sharp, undeniable photo instead of a blurry, inconclusive smudge.

    The goal is to find the right balance—a lens big enough to be confident in the result, but not so big that you waste time and resources.

    ---
    #### What is Minimum Detectable Effect (MDE)? 🔎
    The **Minimum Detectable Effect (MDE)** is the smallest improvement your test can reliably detect at a given power level.

    Think of it as the sensitivity of your experiment. If the true uplift from your change is smaller than the MDE, your test will likely miss it. This doesn't mean the uplift isn't real, just that your experiment isn't powerful enough to see it. Use the MDE to set realistic expectations for what your test can achieve with your available traffic.

    ---
    #### What is Bayesian Power? 💪
    **Power** answers one critical question: *"If my variant is truly better by a specific amount, what's the probability my test will actually detect it?"*

    For example, 80% power means you have an 80% chance of getting a conclusive result (e.g., P(B > A) > 95%) if the real improvement matches what you expected. Running a test with low power is like trying to read in a dim room—you're likely to miss things and end up with an inconclusive result, wasting valuable traffic.

    ---
    #### What are Priors? 🧠
    **Priors** represent what you believe about the conversion rate *before* the test begins. In this model, your belief is captured by two numbers:
    - **Alpha ($$\\alpha$$)**: The number of prior "successes".
    - **Beta ($$\\beta$$)**: The number of prior "failures".

    * **No strong belief?** Use an **uninformative prior** like `alpha = 1` and `beta = 1`. This treats all possible conversion rates as equally likely to start.
    * **Have historical data?** Create an **informative prior**. If past data showed 50 conversions from 1,000 users, you'd set `alpha = 50` and `beta = 950`.

    As your test collects new data, the evidence from the experiment will quickly outweigh the initial prior belief.
    """)
