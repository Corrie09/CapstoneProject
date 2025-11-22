# Capstone Proposal  
**Title:** Bayesian Learning and Frustration in Online Dating Markets  
**Team:** Group 4 — Economics for Data-Driven Decision Making  
**Dataset:** OkCupid Profiles (≈ 60,000 users)

---

## 1. Motivation and Research Question

Online dating platforms are dynamic markets where users face uncertainty about both the type of potential partners they encounter and their own chances of being reciprocated.  
We propose to study **how users update their beliefs** about (i) market composition and (ii) their own attractiveness, and how these evolving beliefs affect **effort and persistence** on the platform.

**Research question:**  
> How do users’ belief updates about the dating market and their own success probability shape their self-presentation and engagement behavior?

Understanding this mechanism matters for platform design: belief-driven frustration can lead to churn, while clearer information can sustain participation.

---

## 2. Theoretical Framework

### 2.1 Setup  
Each user faces two dimensions of uncertainty:

1. **Market composition uncertainty** – fraction of profiles seeking long-term relationships.  
2. **Self-assessment uncertainty** – probability that a “like” will be reciprocated.

Users receive signals through two channels:

- **Viewed profiles:** reveal information about the market’s seriousness.  
- **Responses (or lack thereof):** reveal information about their own acceptance rate.

### 2.2 Bayesian updating (conceptual)
After observing each profile or outcome, users update beliefs:
- Seeing many “short-term” profiles ↓ belief in serious market.  
- Experiencing non-reciprocation ↓ belief in personal success rate.  

Behavioral rule:  
Continue engaging if  
\[
\text{Expected Match Value} \times \text{Expected Acceptance Rate} - \text{Effort Cost} - \text{Rejection Cost} > 0.
\]
When posterior beliefs become pessimistic, effort falls — essays shorten, fields remain empty, or the user exits (“frustration”).

---

## 3. Data Plan

**Dataset:** 60k OkCupid profiles containing demographics (age, sex, orientation, education, income) and 10 text essays.  
Although match outcomes are not observed, cross-sectional patterns in profile effort allow us to proxy belief-driven behavior.

| Theoretical construct | Observable proxy |
|------------------------|------------------|
| Market seriousness belief | Share of users in same location/orientation indicating long-term intent (`status`, `offspring`, essay keywords). |
| Signal precision | Average completeness of profiles viewed (optional fields filled). |
| Effort / engagement | Essay word count, number of optional fields completed. |
| Market tightness (scarcity) | Ratio of opposite-sex to same-sex users within orientation group. |

**Sample construction**
1. Compute market-level statistics by location × orientation group.  
2. Merge these aggregates back to individual profiles.  
3. Estimate relationships between market characteristics and individual effort.

---

## 4. Testable Predictions

1. **Market pessimism effect:** Users in markets dominated by “short-term” profiles display lower essay length and completeness.  
2. **Scarcity effect:** Users in markets where their own type is abundant (high competition) exert less effort.  
3. **Information clarity effect:** Markets with more complete partner profiles show less variance in effort—clear signals slow frustration.

**Empirical specification (conceptual):**
\[
\text{Effort}_{i} = \beta_0 + \beta_1 \text{SeriousShare}_{l} + \beta_2 \text{Scarcity}_{i} + 
\beta_3 (\text{SeriousShare}_{l} \times \text{Scarcity}_{i}) + \mathbf{X}_{i}\gamma + \varepsilon_{i}.
\]

---

## 5. Expected Outputs

| Output | Description |
|---------|--------------|
| **Figure 1** | Histogram of “serious intent” indicators across markets. |
| **Table 1** | Summary statistics of demographics and effort measures. |
| **Figure 2** | Average essay length by quartile of market seriousness. |
| **Table 2** | Regression of effort on market seriousness and scarcity. |
| **Figure 3** | Simulated belief-update curve illustrating frustration dynamics. |

---

## 6. Contributions and Next Steps

This project links **Bayesian learning under uncertainty** to observable user behavior in a matching market.  
It connects the *decision-theory block* (belief updating) with *market-design concepts* (matching and information asymmetry).  
In the coming weeks we will:
1. Construct the “intent” and “effort” indices.  
2. Explore heterogeneity by gender and orientation.  
3. Test whether the data support the predicted belief–effort relationships.

---

**References to course concepts:** Bayesian updating, value of information, asymmetric information, matching equilibrium, and decision under uncertainty.
