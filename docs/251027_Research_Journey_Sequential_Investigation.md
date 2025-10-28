# SwiFT-IO Emotion Prediction: Sequential Scientific Investigation

**Date**: 2025-10-27
**Author**: Research Log
**Status**: Ongoing Investigation

---

## 🔬 Executive Summary

This document chronicles the scientific investigation of SwiFT-IO's emotion prediction capability, from initial promising metrics to deep systematic validation. The journey reveals critical insights about the gap between quantitative metrics and actual model performance, the challenges of interpretability, and the importance of rigorous validation protocols.

**Key Finding**: What appeared as strong performance in aggregate metrics masked fundamental issues revealed only through visualization and interpretability analysis. The shift to binary classification exposed dramatic performance disparities that are currently under systematic validation.

---

## Phase 1: Initial Performance Assessment - The Metric-Reality Gap

### Initial Hypothesis
SwiFT-IO outperforms baseline models (SVR, LSTM, GLM) in emotion prediction from fMRI.

### Experimental Setup
- **Task**: Regression on continuous emotion intensities
- **Comparison**: SwiFT-IO vs. multiple baselines
- **Metrics**: MSE, MAE, R², Pearson correlation

### Results

**Quantitative Assessment** (Aggregate Metrics):
```
SwiFT-IO performance > Baseline models
MSE ↓, MAE ↓, R² ↑ compared to all baselines
```
✅ Initial conclusion: SwiFT-IO is superior

**Qualitative Assessment** (Frame-by-frame visualization):
```
Model predictions:
- Overly smooth
- Missing temporal dynamics
- Failing to capture emotion peaks
- Regressing toward mean
```
❌ Visual inspection: Poor actual prediction quality

### Critical Finding: The Metric-Reality Discrepancy

**Observation**: Aggregate metrics showed good performance, but frame-by-frame plots revealed the model was essentially predicting smooth averages rather than capturing true emotional dynamics.

**Scientific Question**:
> "Why do aggregate metrics suggest success while visual inspection suggests failure?"

**Hypotheses**:
1. **Regression formulation issue**: Sparse, imbalanced labels not suited for regression
2. **Model limitation**: Architecture insufficient for temporal modeling
3. **Evaluation metric mismatch**: Metrics rewarding average prediction rather than peak capture

### Implications
- Aggregate metrics alone are insufficient for model evaluation
- Visualization is critical for understanding actual model behavior
- Need to investigate model's internal representations

---

## Phase 2: Interpretability via Integrated Gradients - The Baseline Problem

### Motivation
Understand which brain regions the model actually uses for emotion prediction.

### Experiment 2.1: IG with Baseline = 0

**Method**:
```python
IG(x) = (x - baseline) × ∫[α=0→1] ∂f(baseline + α(x-baseline))/∂x dα

Where baseline = zeros (shape of fMRI volume)
```

**Results**:
- High attribution in visual cortex
- **Problem**: Attribution outside brain boundaries (anatomically implausible)
- **Problem**: Only visual areas shown as important (neuroscientifically insufficient for emotion)

**Interpretation Issues**:
- fMRI signals are all positive → entire activation interpreted as contribution
- Non-brain voxel noise also contributes when far from zero
- Baseline = 0 is not a meaningful reference state for fMRI

### Experiment 2.2: IG with Baseline = First 10 TRs

**Rationale**: Use movie onset period as "neutral baseline" (more realistic reference state)

**Results**:
✅ **Improvement**: Non-brain attribution resolved
❌ **New Problem**: IG values extremely small overall
❌ **Critical Issue**: **No spatial pattern differences across emotions**
❌ **Interpretation**: Even top 5% voxels show uninformative patterns

### Mechanistic Analysis: Why IG Failed

**The Gradient Problem**:
```
If model output ≈ baseline for most inputs:
→ ∂f/∂x ≈ 0
→ IG ≈ 0
→ No attribution signal
```

**Root Cause Hypothesis**:
The regression model learned to output near-constant predictions (close to mean), making gradients vanishingly small regardless of input variation.

### Key Insight
> "Small IG values indicate the model is not strongly utilizing input features"
> "Lack of emotion-specific spatial patterns suggests failure to learn meaningful emotional representations"

### Scientific Question
> "Is the regression loss function preventing the model from learning spatially-specific, emotion-discriminative patterns?"

---

## Phase 3: Task Reformulation - Classification as Diagnostic Tool

### Strategic Pivot

**Rationale**:
- Regression showing poor interpretability and temporal dynamics
- Binary classification is simpler → easier to diagnose problems
- Sparse labels (mostly zeros) may be more suited to classification
- Classification requires learning presence/absence → clearer decision boundary

### Experimental Design

**Task Transformation**:
```
Regression: Predict continuous emotion intensity [0, 1]
            ↓
Classification: Predict binary label {0: absent, 1: present}
                (threshold original labels at 0)
```

**Comparison Variables**:
- **Sequence length**: 20 TRs (26s) vs. 30 TRs (39s)
- **Architecture**: Same SwiFT-IO backbone
- **Output head**: Binary classification head (2 classes per emotion)

### Results

#### Seq 20 Performance:
```
AUROC Range: 0.66 - 0.81
Best: Sad (0.81), Fear (0.67)
Worst: Positive (0.66, but most balanced)

Coverage: 98.7% of 750 frames
```
✅ Generally good discrimination

#### Seq 30 Performance:
```
AUROC Range: 0.57 - 0.65  (near random for most)

With threshold = 0.5:
- Anger, Happy, Fear, Sad, Excited, Negative: Recall = 0.0
- Complete collapse to majority class (predict all zeros)

Coverage: 100% of 750 frames
```
❌ Catastrophic failure

### The Dramatic Disparity

**Observation**: 10 TR difference (13 seconds) causes complete performance collapse

**Initial Reaction**: Suspicion
- Seq 20 seems "too good"
- Seq 30 seems "too bad"
- Difference seems mechanistically implausible

**Red Flags**:
1. **Magnitude**: AUROC drop from 0.8 to 0.6 is huge
2. **Abruptness**: Binary classification should be robust
3. **Coverage vs Performance**: Seq 30 has better coverage but worse performance

---

## Phase 4: Critical Examination - The Youden Threshold Discovery

### The Triggering Observation

**Plot Analysis**: Probability outputs range continuously from 0 to 1, not binary predictions.

**Question**: "How is the classification threshold determined?"

### Code Investigation

**Finding**:
```python
# In pl_classifier.py line 371 (original):
predictions = probabilities.argmax(dim=-1)

# This is equivalent to:
predictions = (probabilities[:, 1] >= 0.5).long()
```

**Implication**: **Fixed threshold = 0.5 for all emotions**

### Critical Realization

**Standard Practice in Imbalanced Classification**:
Use **Youden Index** to find optimal threshold:
```
J = Sensitivity + Specificity - 1 = TPR - FPR

Optimal threshold = argmax_t (J(t))
```

**Why This Matters**:
- Emotion labels are highly imbalanced (80-90% zeros for most)
- Threshold = 0.5 is arbitrary and suboptimal for imbalanced data
- Seq 30's "collapse" might be **evaluation artifact** not true failure

### Validation of Youden Threshold

**Implementation**:
1. Find optimal threshold on validation set per emotion
2. Apply to test set
3. Compare threshold = 0.5 vs optimal

**Results** (`analysis/optimal_thresholds.json`):

**Seq 20 Optimal Thresholds**:
```
Sad:      0.088 (highly imbalanced)
Anger:    0.284
Excited:  0.214
Positive: 0.504 (most balanced)
Happy:    0.141
Fear:     0.197
Negative: 0.249
```

**Seq 30 Optimal Thresholds**:
```
Sad:      0.091
Anger:    0.296
Excited:  0.234
Positive: 0.517
...
```

**Key Finding**:
With threshold = 0.5, Seq 30 predicts **recall = 0** for most emotions.
With optimal thresholds, performance is **rescued** (though still worse than Seq 20).

### Impact on Interpretation

**Before Youden**:
- Seq 30 appears completely broken
- Suggests fundamental model failure

**After Youden**:
- Seq 30 learns reasonable probabilities
- Problem is **evaluation**, not learning
- Performance gap remains but is less dramatic

---

## Phase 5: Validation Protocol - Systematic Doubt

### Guiding Principle

> "Extraordinary claims require extraordinary evidence"

**Current Claim**: Seq 20 dramatically outperforms Seq 30

**Skepticism Warranted Because**:
1. Performance difference is very large
2. Mechanism is unclear (why would 10 TRs matter so much?)
3. Both "too good" (Seq 20) and "too bad" (Seq 30 with thr=0.5) raise flags

### Validation Strategy: Three-Pronged Approach

#### Validation 1: Best vs. Last Checkpoint Analysis

**Job ID**: 63384
**Status**: Running
**Duration**: ~30 minutes

**Hypothesis**: Seq 30 stopped improving early, but we evaluated wrong checkpoint

**Observations**:
- Seq 20 best checkpoint: Epoch 8 (valid_acc = 1.00)
- Seq 30 best checkpoint: Epoch 3 (valid_acc = 0.72)
- Seq 30 last checkpoint: Epoch 24 (currently evaluated)

**Test**: Compare Seq 30 epoch 3 vs. epoch 24 on test set

**Interpretation Guide**:
- **If Best >> Last**: Model overfit after epoch 3 → training dynamics problem
- **If Best ≈ Last**: No overfitting → fundamental capacity limitation
- **If Best < Last**: Something else is wrong (investigate further)

**Scientific Value**: Determines whether collapse is training issue or model issue

---

#### Validation 2: Class Imbalance Analysis

**Job ID**: 63389
**Status**: Running
**Duration**: ~1 hour

**Hypothesis**: Longer sequences amplify class imbalance

**Mechanism**:
```
Seq 20: N sequences × 20 frames = Total samples
Seq 30: M sequences × 30 frames = Total samples
(M < N due to stride, but similar total samples)

Question: Does distribution of positives differ?
```

**Metrics to Compare**:
- Positive ratio per emotion
- Imbalance ratio (negative:positive)
- Change in imbalance from Seq 20 → Seq 30

**Test**:
```
For each emotion:
  Compute: Imbalance_20, Imbalance_30
  Calculate: ΔImbalance = Imbalance_30 - Imbalance_20
```

**Interpretation Guide**:
- **If ΔImbalance > 1**: Seq 30 has worse imbalance → **data distribution problem**
- **If ΔImbalance ≈ 0**: Similar imbalance → **architecture/training problem**

**Possible Outcomes**:

**Scenario A: Data Distribution Hypothesis**
```
Longer sequences → more consecutive zero frames → worse imbalance
→ Model learns to predict all zeros (majority class)
→ Explanation: Statistical not architectural
```

**Scenario B: Architecture Hypothesis**
```
Imbalance similar across seq lengths
→ Problem is model's ability to handle longer sequences
→ Possible causes:
  - Gradient flow issues
  - Attention span limitations
  - Memory requirements
```

**Scientific Value**: Distinguishes data vs. model causes

---

#### Validation 3: Random Seed Replication

**Job ID**: 63388
**Status**: Running
**Duration**: ~1-2 days (full training)

**Hypothesis**: Performance difference is reproducible across random seeds

**Setup**:
```
Original training: seed = 777 → collapsed
Replication:       seed = 888 → ?
```

**Controlled Variables**:
- Architecture: Same
- Hyperparameters: Same
- Data split: Different (seed controls split)
- Weight initialization: Different

**Interpretation Guide**:

**Scenario A: Reproducible Collapse**
```
Seed 888 also collapses (similar poor performance)
→ Systematic issue
→ Not due to unlucky initialization
→ Likely: Architecture, hyperparameters, or data issue
```

**Scenario B: Seed-Dependent Performance**
```
Seed 888 works well (similar to Seq 20)
→ Original collapse was bad luck
→ Initialization sensitive
→ Problem: Training instability
```

**Scenario C: Partial Replication**
```
Seed 888 intermediate performance
→ High variance across seeds
→ Suggests: Optimization landscape issues
→ Need: More runs to characterize distribution
```

**Scientific Value**:
- Tests reproducibility (cornerstone of science)
- Reveals initialization sensitivity
- Informs whether problem is deterministic or stochastic

---

### Synthesis of Validation Results

**After all three validations complete**, triangulate findings:

```
              ┌─────────────────┐
              │  Best < Last?   │
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         │ YES                      │ NO
         ▼                          ▼
    Overfitting                 Capacity
    problem                     problem
         │                          │
         └─────────┬────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Imbalance worse?   │
         └─────────┬──────────┘
                   │
         ┌─────────┴──────────┐
         │ YES              │ NO
         ▼                  ▼
    Data issue        Architecture
                      or training
         │                  │
         └────────┬─────────┘
                  │
         ┌────────▼─────────┐
         │ Seed replicates? │
         └────────┬─────────┘
                  │
         ┌────────┴─────────┐
         │ YES            │ NO
         ▼                ▼
    Systematic      Initialization
    failure         sensitivity
```

---

## Scientific Thinking Framework

### 1. The Iterative Hypothesis Cycle

```
H₀: SwiFT-IO > baselines
├─ Test: Aggregate metrics
├─ Result: Metrics support H₀
├─ Counter-evidence: Visualizations don't support
└─ Decision: Reject H₀, investigate mechanism

H₁: Model interpretability will reveal mechanism
├─ Test: IG maps with baseline=0
├─ Result: Artifacts and poor signal
├─ Refinement: IG maps with baseline=10TRs
├─ Result: Better anatomically, still uninformative
└─ Decision: Reject H₁, try different approach

H₂: Binary classification will clarify regression issues
├─ Test: Classification task on same data
├─ Result: Seq 20 good, Seq 30 bad
└─ Decision: New mystery → need more evidence

H₃: Threshold choice matters critically
├─ Test: Optimal vs. fixed threshold
├─ Result: Fixed threshold causes false negatives
└─ Decision: Partial explanation, but gap remains

H₄: Performance difference is real and systematic
├─ Test: 3-way validation (checkpoint/imbalance/seed)
├─ Status: In progress
└─ Decision: Pending results
```

### 2. Multi-Level Evidence Integration

**Principle**: Converging evidence from multiple methods strengthens conclusions

**Evidence Pyramid**:
```
                    🔺
                   /  \
                  / 📊 \
                 /Quant \
                /________\
               /          \
              /     📈     \
             /   Visual    \
            /______________\
           /                \
          /       🧠         \
         /   Mechanistic     \
        /____________________\
       /                      \
      /         🔄            \
     /     Replication         \
    /__________________________\
```

**Our Application**:
1. **Quantitative**: AUROC, accuracy, F1 scores
2. **Visual**: Time-series plots, heatmaps, IG maps
3. **Mechanistic**: Understanding why models fail/succeed
4. **Replication**: Multiple seeds, conditions

**Integrated Assessment**:
Only when all levels agree should we have confidence

### 3. Epistemic Humility: Knowing What We Don't Know

#### What We Know (High Confidence):

✅ **Regression produces uninformative predictions**
- Evidence: Visual inspection, IG maps
- Replication: Consistent across attempts
- Mechanism: Understood (regressing to mean)

✅ **IG baseline choice matters critically**
- Evidence: Anatomical plausibility changes
- Mechanism: Understood (gradient calculation)

✅ **Classification threshold affects evaluation**
- Evidence: Dramatic metric changes
- Mechanism: Understood (imbalance statistics)
- Implementation: Fixed in code

✅ **Seq 20 and Seq 30 show different behaviors**
- Evidence: Consistent in initial experiments
- Replication: Needs validation (in progress)

#### What We Think We Know (Medium Confidence):

⚠️ **Binary classification works for Seq 20**
- Evidence: AUROC 0.66-0.81
- Concern: Seems too good, needs validation
- Test: In progress (seed replication)

⚠️ **Seq 30 has capacity issues**
- Evidence: Poor performance, early stopping
- Alternative: Could be data, hyperparameter, or luck
- Test: In progress (all three validations)

#### What We Don't Know (Low Confidence):

❓ **Why Seq 20 vs. Seq 30 differ so dramatically**
- Hypotheses: Architecture/data/training/luck
- Tests: In progress
- Resolution: Awaiting validation results

❓ **Whether SwiFT-IO is actually good for emotions**
- Status: Original question still unresolved
- Blockers: Need to validate Seq 20 first
- Next: If Seq 20 validated, compare to stronger baselines

❓ **Optimal sequence length for this task**
- Current: Only tested 20 and 30
- Needed: Sweep [15, 20, 25, 30, 35, 40]
- Goal: Find critical transition point

❓ **Whether classification or regression is better**
- Current: Classification seems better
- Caveat: Might be task/data specific
- Needed: Principled comparison on validated model

---

## Lessons Learned: Meta-Scientific Insights

### 1. Metrics Lie, Visualizations Reveal

**The Trap**:
Aggregate metrics (MSE, MAE, R²) showed good performance
→ Easy to publish, stop investigating
→ Disaster waiting to happen

**The Reality**:
Frame-by-frame visualization revealed:
- Model predicting smooth averages
- Missing all temporal dynamics
- Uninformative for scientific understanding

**Takeaway**:
> "Never trust a number you haven't plotted"

### 2. Interpretability Is Hard (And Often Uninformative)

**The Hope**:
IG maps will show which brain regions drive emotion prediction
→ Neuroscientific insight
→ Validate model is using right areas

**The Reality**:
- Baseline choice critically affects results
- Small gradients → small attributions
- Emotion-specific patterns absent

**Takeaway**:
> "Interpretability methods can fail silently"
> "Lack of interpretable patterns is itself a signal"

### 3. Simpler Tasks Can Reveal Complex Problems

**The Strategy**:
Regression struggling → Try binary classification
→ Simpler task should be easier
→ Diagnostic value

**The Surprise**:
Classification revealed NEW mysteries:
- Dramatic Seq length sensitivity
- Threshold selection criticality
- Needs as much validation as regression

**Takeaway**:
> "Simplification can expose rather than resolve complexity"

### 4. Details Matter Catastrophically

**The Detail**:
Using `argmax()` vs. optimal threshold
→ Seems like implementation choice
→ "Surely it's close to optimal?"

**The Impact**:
- Difference between "complete failure" and "moderate success"
- Changes scientific interpretation entirely
- Could have led to wrong conclusions

**Takeaway**:
> "In imbalanced classification, threshold = 0.5 is not a 'reasonable default'"
> "Always validate assumptions that seem 'obvious'"

### 5. Healthy Skepticism of Positive Results

**The Temptation**:
Seq 20 shows AUROC 0.8 → Great, publish!

**The Red Flags**:
- Seems too good compared to Seq 30
- Seq 30 collapse seems too dramatic
- Mechanism unclear

**The Response**:
- Systematic validation protocol
- Multiple lines of evidence
- Replication before belief

**Takeaway**:
> "Extraordinary results require extraordinary scrutiny"
> "Be suspicious when things look too good OR too bad"

### 6. The Reproducibility Crisis Is Real

**The Standard**:
Run once, get result, interpret, publish

**The Problem**:
- Different seeds can change everything
- Initialization matters more than we admit
- Single runs can be misleading

**Our Approach**:
- Explicit seed replication
- Testing best vs. last checkpoint
- Multiple validation axes

**Takeaway**:
> "If it's not reproducible, it's not real"
> "Build validation into the workflow, not after"

---

## Current Status: Living at the Edge of Knowledge

### What We're Doing Right Now

**Active Validations** (2025-10-27):

```
┌─────────────────────────────────────────┐
│ Job 63384: Best Checkpoint Eval         │
│ Status: Running (4+ hours)              │
│ ETA: Soon                               │
│ Will Tell: Training dynamics issue?     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Job 63389: Class Imbalance Analysis     │
│ Status: Running (1+ hour)               │
│ ETA: Soon                               │
│ Will Tell: Data distribution issue?     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Job 63388: Seed 888 Training            │
│ Status: Running (< 1 hour so far)       │
│ ETA: 1-2 days                           │
│ Will Tell: Reproducibility?             │
└─────────────────────────────────────────┘
```

### The Decision Tree Ahead

**Scenario 1: Validation Confirms Issues**
```
→ Seq 30 has fundamental problems (data or architecture)
→ Focus on understanding Seq 20
→ If Seq 20 validates, proceed with publication
→ Include Seq 30 as negative result for completeness
```

**Scenario 2: Validation Invalidates Seq 20**
```
→ Seq 20 was luck or artifact
→ Back to drawing board
→ Need to fundamentally rethink approach
→ Possibly return to regression with new insights
```

**Scenario 3: Mixed Results**
```
→ Some validations pass, some fail
→ More targeted investigations needed
→ Possibly: Seq length sweep [15, 20, 25, 30, 35]
→ Identify critical transition point
```

**Scenario 4: Everything Works**
```
→ Seq 20 validates, mechanism clear
→ Proceed to publication
→ Include full validation story
→ Emphasize importance of proper evaluation
```

---

## Broader Implications

### For This Project

**If Successful**:
- Demonstrates SwiFT-IO can predict emotions from fMRI
- Provides optimal sequence length guidance
- Highlights importance of proper threshold selection
- Creates validated pipeline for future work

**If Unsuccessful**:
- Rules out naive classification approach
- Informs future architecture designs
- Valuable negative result for community
- Redirects effort toward more promising directions

### For Neuroimaging ML Generally

**Methodological Contributions**:
1. **Visualization as validation**: Aggregate metrics insufficient
2. **Interpretability skepticism**: IG maps not always helpful
3. **Threshold optimization**: Critical for imbalanced neuroscience data
4. **Replication protocols**: How to validate neuroimaging models

**Cautionary Tales**:
1. Don't trust metrics alone
2. Baseline choice in IG matters enormously
3. Classification not always simpler than regression
4. Sequence length can have dramatic non-obvious effects

### For Scientific Practice

**Process Insights**:
- Iterative hypothesis testing works
- Multi-modal evidence is essential
- Skepticism of own results is healthy
- Validation should be built-in, not retrofitted

**Cultural Insights**:
- Positive results need scrutiny too
- Negative results have scientific value
- Reproducibility should be default
- Uncertainty should be embraced

---

## Next Steps (When Validations Complete)

### Immediate (Within 1 Week)

1. **Analyze validation results**
   - Integrate findings from all three validations
   - Update interpretation based on evidence
   - Decide on primary hypothesis

2. **Generate comprehensive report**
   - Performance tables with optimal thresholds
   - Visualization of all results
   - Statistical tests of differences

3. **Make go/no-go decision**
   - Proceed with publication if validated
   - Return to development if invalidated

### Short-Term (Within 1 Month)

**If Validated**:
1. Sequence length sweep [15, 20, 25, 30, 35, 40]
2. Additional seed replications (seed 999, 1000, 1001)
3. Comparison to stronger baselines
4. Manuscript preparation

**If Not Validated**:
1. Diagnosis of failure mode
2. Architecture modifications
3. Hyperparameter optimization
4. Alternative task formulations

### Long-Term (Within 3 Months)

**If Validated**:
1. Apply to other movie datasets
2. Generalization to different emotions/tasks
3. Mechanistic interpretation (if possible)
4. Clinical applications

**If Not Validated**:
1. Fundamental rethinking
2. Possible pivot to different approach
3. Focus on what we learned
4. Negative result publication

---

## Conclusion: The Value of Rigorous Investigation

### What This Journey Taught Us

**Scientific Virtues Demonstrated**:
- ✅ Questioning positive results
- ✅ Multi-level validation
- ✅ Visualization over blind metrics
- ✅ Replication as standard
- ✅ Epistemic humility

**Common Pitfalls Avoided**:
- ❌ Accepting metrics at face value
- ❌ Publishing without validation
- ❌ Ignoring visualizations
- ❌ Over-interpreting single results
- ❌ Assuming reproducibility

### Why This Matters

> "Science is not about being right, it's about getting less wrong over time"

This investigation exemplifies:
1. **Iterative refinement**: Each phase revealed new questions
2. **Healthy skepticism**: Question everything, including successes
3. **Multiple evidence**: Triangulation from many angles
4. **Reproducibility**: Test it before you trust it
5. **Transparency**: Document the journey, not just the destination

### The Path Forward

**Regardless of final outcome**, this investigation has value:
- Methodological insights for neuroimaging ML
- Demonstration of rigorous validation
- Cautionary tale about metrics
- Template for future investigations

**The scientific method in action**:
```
Observe → Hypothesize → Test → Doubt → Refine → Repeat
                                  ↑
                          We are here
```

We await validation results to determine the next iteration of this cycle.

---

## Appendix: Technical Details

### Code Modifications Made

1. **Optimal Threshold Implementation**
   - Location: `src/module/pl_classifier.py`
   - Function: `_calculate_optimal_thresholds()`
   - Integration: Called in `validation_epoch_end()`
   - Applied: In `_evaluate_metrics()` for test set

2. **Visualization Enhancements**
   - File: `compare_classification_seq20_seq30.py`
   - Added: Optimal threshold lines to plots
   - Added: Binary correctness heatmap
   - Output: `analysis/plots/classification_seq20_vs_seq30/`

3. **Analysis Scripts Created**
   - `find_optimal_thresholds.py`: Threshold calculation
   - `analyze_class_imbalance.py`: Data distribution analysis
   - `evaluate_best_checkpoint.py`: Checkpoint comparison
   - `train_seq30_seed888.slurm`: Replication training

### Data Files Generated

1. **Optimal Thresholds**: `analysis/optimal_thresholds.json`
2. **Plots**: `analysis/plots/classification_seq20_vs_seq30/*.png`
3. **Job Outputs**: `*-6338[4,5,6,7,8,9].out/err`

### Experiments In Progress

| Experiment | Job ID | Status | Expected Completion |
|------------|--------|--------|---------------------|
| Best Checkpoint Eval | 63384 | Running | 2025-10-27 PM |
| Class Imbalance | 63389 | Running | 2025-10-27 PM |
| Seed 888 Training | 63388 | Running | 2025-10-29 |

---

**Document Status**: Living document, will be updated as results arrive
**Last Updated**: 2025-10-27
**Next Update**: Upon validation completion
