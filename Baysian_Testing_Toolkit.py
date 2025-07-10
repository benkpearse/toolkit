import streamlit as st

st.set_page_config(page_title="Bayesian Testing Toolkit", layout="centered")

st.title("📊 Bayesian Testing Toolkit")

st.markdown("""
Welcome to the **Bayesian Testing Toolkit**. This suite includes three apps designed to support high-quality, data-informed A/B testing:

---

### 1. 📈 Uplift Certainty Estimator
Use this to interpret results after an A/B test has run.

- **Input:** Number of users and conversions for both variants.
- **Output:**
  - Probability that Variant B is better than A
  - Estimated uplift
  - Credible interval
  - Graphs of posterior distributions
  - Sample Ratio Mismatch check - Please input User not Session data to identify SRM

🧠 **Use when:** 
- During testing to evaluate results.
- Your test is complete and you want to evaluate results with a Bayesian lens.

---

### 2. ⚙️ Sample Size Calculator
Estimate the minimum **sample size per variant** needed to detect an uplift with a certain confidence.

- **Input:** Baseline rate, expected uplift, probability target, confidence threshold, optional priors
- **Output:** Minimum sample size or power achieved for given n

🧠 **Use when:**
- You're **planning** a test and want to ensure it's properly powered. You can use **priors from past experiments or pre-test data** to inform assumptions.
- You're **midway through a test** and want to check whether you're likely to reach significance with current progress (using real-time priors).
- You've **completed a test** and want to validate whether the test had enough data to justify the conclusion.

---

### 3. 🚨 False Positive Estimator
Simulate the **false positive rate** of your test setup when there's no real difference between A and B.

- **Input:** Conversion rate, confidence threshold, sample size
- **Output:** Estimated false positive rate

🧠 **Use when:** You want to **validate the reliability** of your decision threshold before launching a test.

---

👉 Select a page from the **sidebar** to get started.
""")
