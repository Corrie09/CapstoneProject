# AI Coding Agent Instructions for OkCupid Dating Market Analysis

This capstone project empirically tests economic theories of dating market behavior, specifically the "frustration hypothesis" - that users lower their effort when markets are unfavorable.

## Project Architecture

**Data Processing Pipeline** (6 sequential steps, each in `notebooks/` or `EDA/`):
1. **Step 1** (`EDA/step1_data_exploration.py`): Load raw OkCupid data, profile info
2. **Step 2** (`notebooks/step2_data_cleaning.py`): Standardize categorical vars (education→ordinal, income→clean, etc.)
3. **Step 3** (`notebooks/step3_effort_index.py`): Compute effort index from essay word counts, completeness
4. **Step 4** (`notebooks/step4_rating_index.py`): Build desirability score using education/income/height/age/body_type
5. **Step 5** (`notebooks/step5_relationship_goals.py`): Extract LTR vs casual orientation, create binary indicators
6. **Step 6** (`notebooks/step6_market_indices.py`): Aggregate to market level (location × sex × orientation)

**Output progression**: Raw CSV → cleaned → with_effort → with_ratings → with_goals → final_analysis_ready
- All intermediate outputs stored in `notebooks/outputs/`
- Final data for analysis: `notebooks/outputs/Final/okcupid_final_analysis_ready.csv`

**Economic Simulation Layer** (`simulations/`):
- Extract empirical parameters from data via `extract_parameters.py` → `simulation_parameters.json`
- Run Bayesian learning simulations in `exit_time_simulations.py` to predict user exit behavior
- Framework models users with priors on market composition (ρ), testing when users quit searching

## Key Concepts & Conventions

**Core Variables** (appear consistently across all scripts):
- `effort_index` (E_i): Composite of essay words + essays completed + profile completeness
- `rating_index` (r_i): Normalized [0,1] score of user desirability (higher education/income/body shape weighted)
- `is_ltr_oriented`: Binary indicator (True if user seeks long-term relationships)
- `market_id`: Unique ID combining location + sex + orientation_of_searcher
- `target_market`: Market where user is *searching*, may differ from their own demo

**Data Flow Assumptions**:
- Each script expects its input file to exist and exits with helpful error message if missing
- **File paths are relative** to project root: `data/`, `notebooks/`, `simulations/`
- All scripts output to `notebooks/outputs/` with subdirectories: `summary/` (text summaries), `Final/` (analysis-ready data)

**Normalization Conventions**:
- Rating index always normalized to [0,1] via MinMaxScaler after component aggregation
- Effort index normalized by z-score within market groups (not global)
- Categorical variables converted to ordinal (education: 0–5, income: log-scaled)

## Simulation Framework

The economic model tests the frustration hypothesis by simulating Bayesian learners:
- **Prior**: Users start with belief about LTR market share (ρ_i0), strength τ_i
- **Observation**: Each time step, draw K=20 potential matches, count "successes" (LTR-oriented)
- **Update**: Update posterior belief via Beta-Binomial conjugacy
- **Exit Rule**: User quits if posterior belief < p_bar (frustration threshold ~0.20)

Key parameters extracted from data (in `simulation_parameters.json`):
- `alpha_rating`, `beta_rating`: Beta distribution shape parameters for rating distribution
- `rho_i0_ltr`, `tau_i_suggested`: Self-based prior and confidence for LTR users
- `K`: Batch size (typically 20 matches per period)

## Common Workflows

**Adding new market segment analysis**:
1. Filter `okcupid_final_analysis_ready.csv` by market criteria
2. Calculate new `market_id` if needed (location + sex + orientation)
3. Recompute market indices in Step 6 logic (LTR share, clarity, competition)
4. Re-run simulations with segment-specific parameters

**Extending effort/rating metrics**:
- Add component in Step 3 (essay metrics) or Step 4 (rating components)
- Normalize within market groups to preserve comparative advantage
- Recompute downstream steps (aggregate data must be regenerated)

**Testing new hypothesis**:
1. Verify data has required variables (check Step 6 output)
2. Compute summary statistics by market using `.groupby(['market_id'])`
3. Run regression: effort ~ market_seriousness + other controls
4. Cross-validate with simulation predictions from `exit_time_simulations.py`

## File Structure Notes

- **`data/OLD/`**: Deprecated versions of raw dataset (ignore for new analysis)
- **`simulations/BROL/`**: Experimental/test scripts (use only as reference)
- **`slides/BROL/`**: Draft slide versions (check `newestFull.pdf` for latest presentation)
- **Key reference outputs**: 
  - `notebooks/outputs/Final/okcupid_final_analysis_ready.csv` (master dataset)
  - `notebooks/outputs/summary/` (text summaries of each step)
  - `simulations/simulation_parameters.json` (empirically calibrated model params)

## Dependencies & Execution

All packages in `pyproject.toml`: pandas, numpy, matplotlib, seaborn, scikit-learn, sweetviz
- Python ≥3.13 required
- Scripts run standalone as `python step_X.py` (not installed as package)
- No tests defined; validate via intermediate CSV shape/columns

## Debugging Strategy

If a step fails:
1. Check file path in error message against actual `data/` or `notebooks/outputs/` location
2. Verify all previous steps completed (missing upstream CSV = missing dependency)
3. Check hardcoded paths match workspace (many scripts use relative paths like `'data/okcupid_profiles.csv'`)
4. Inspect intermediate outputs: shape, column names, null counts in summary files
