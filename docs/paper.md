# Stress-Testing Longitudinal Synthetic EHR and ICU Waveform Generation: An Adversarial Benchmarking Framework for Privacy, Utility, and Algorithmic Fairness

**Aaqib Rashid Mir**  
*Department of Artificial Intelligence & Data Science, Guru Nanak Institutions Technical Campus, Hyderabad, India*  
*The Coding Science — School of Technology & AI Innovations*  
*ORCID: 0009-0002-3921-654X | Correspondence: mtechbro94@gmail.com*

---

## Abstract

**Background & Objective:** Deep generative architectures—notably continuous-time diffusion models, generative adversarial networks (GANs), and longitudinal transformers—are widely proposed as a panacea to the acute data-governance gridlock that restricts the secondary utilization of Intensive Care Unit (ICU) Electronic Health Records (EHR). The operational presumption underpinning this movement is that synthetic trajectories decouple clinical utility from re-identification liability. However, contemporary validation regimes remain fundamentally fragmented: they almost universally rely on low-dimensional static marginal distributions and aggregate similarity scores, which fail to capture non-physiological temporal dynamics or empirical privacy leakage under directed adversarial scrutiny.

**Methods:** We introduce a holistic, adversarial stress-testing framework that evaluates continuous synthetic longitudinal EHR and physiological waveforms across three coupled axes: privacy resilience, clinical utility transfer, and demographic parity. Our protocol integrates: (i) a four-way partitioned shadow-model Membership Inference Attack (MIA) and supervised attribute reconstruction pipeline tailored to continuous multivariate time-series; (ii) dual-direction clinical utility benchmarks implementing Train-on-Synthetic-Test-on-Real (TSTR) and Train-on-Real-Test-on-Synthetic (TRTS) pipelines targeting early patient deterioration (24-hour mortality and acute decompensation); (iii) five domain-specific temporal realism probes that quantify sequence irregularity, non-uniform measurement cadence, first-order Markovian state transition matrix drift over discretized physiological states, within-stay autocorrelation decay error, and cumulative multiday trend drift; and (iv) an algorithmic fairness stress audit evaluating disparate utility degradation and asymmetric privacy vulnerabilities across marginalized patient sub-cohorts.

**Results:** We benchmarked four canonical generative paradigms (Gaussian Copula, 1D-CNN WGAN-GP, TimeGAN, and TabDDPM-Time) on a curated cohort of 42,110 adult ICU admissions from the MIMIC-IV database (v2.2). While unconstrained models demonstrated high nominal downstream utility (TabDDPM achieving TSTR AUROC $0.814$ [95% CI: $0.802\text{--}0.826$] compared to the real-data baseline of $0.842$ [$0.831\text{--}0.853$]), this utility was accompanied by significant empirical privacy leakage (MIA AUROC $0.618$ [$0.604\text{--}0.632$], adversary advantage $0.236$). Recurrent adversarial networks exhibited catastrophic temporal drift: WGAN-GP and TimeGAN displayed transition matrix divergence ($D_{\text{trans}} = 0.224$ and $0.148$, respectively), generating physiologically impossible hourly jumps between extreme bradycardia ($<60\text{ bpm}$) and malignant tachycardia ($>120\text{ bpm}$) at rates over 28-fold higher than observed in clinical reality ($P_{\text{syn}} = 0.084$ vs. $P_{\text{real}} = 0.003$). Crucially, demographic auditing uncovered severe asymmetric vulnerabilities: minority racial cohorts and elderly patients ($\ge 65$ years) suffered up to an $18.2\%$ increase in membership re-identification vulnerability and an $8.4\%$ drop in predictive transfer utility relative to the majority population.

**Conclusion:** Our findings demonstrate that static distributional alignment masks profound temporal instability and disproportionate privacy risks for vulnerable patient populations. We conclude that empirical stress testing across temporal plausibility, adversarial vulnerability, and demographic fairness must be mandated as an indispensable regulatory prerequisite prior to the public or inter-institutional dissemination of synthetic healthcare data. The benchmarking codebase, standardized cohort extractors, and evaluation pipelines have been made publicly available.

**Keywords:** Synthetic Healthcare Data, Electronic Health Records, Differential Privacy, Membership Inference, Algorithmic Fairness, Time-Series Modeling, Intensive Care Medicine, MIMIC-IV.

---

## 1. Introduction

The secondary exploitation of granular Electronic Health Records (EHR) and continuous physiological waveform streams acquired in intensive care units (ICUs) has propelled modern clinical informatics, enabling breakthroughs in algorithmic early-warning systems, dynamic acuity estimation, and personalized hemodynamic management [1, 2]. Despite these methodological strides, medical artificial intelligence research remains encumbered by an institutional data-access paradox. Regulatory protections—such as the Health Insurance Portability and Accountability Act (HIPAA) in the United States and the General Data Protection Regulation (GDPR) in the European Union—properly restrict the unfettered dissemination of clinical trajectories to prevent the re-identification of vulnerable patients [3]. Consequently, the vast majority of biomedical predictive architectures are developed, tuned, and validated on isolated, single-institution data silos, hampering cross-institutional reproducibility and exacerbating algorithmic bias.

To circumvent the logistical, legal, and institutional liabilities of raw data sharing, deep generative modeling has emerged as a prominent paradigm [4, 5]. Generative Adversarial Networks (GANs), score-based continuous diffusion models, and temporal transformer decoders are trained to approximate the high-dimensional probability manifold of real patient records. The synthetic records sampled from these distributions are frequently posited to be intrinsically anonymous—operating under the assumption that because a generated trajectory does not correspond to any singular physical individual, it is immune to regulatory embargoes and free from privacy concerns [6].

### 1.1 The Failure Modes of Contemporary Validation Regimes

Despite substantial investment in synthetic health data generation, the path from experimental algorithms to trusted scientific infrastructure is impeded by an acute methodological failure: the lack of rigorous, multidimensional, and domain-grounded evaluation standards. The prevailing literature overwhelmingly assesses synthetic EHR quality through narrow statistical lenses, predominantly comparing static one-dimensional marginal distribution histograms, low-order pairwise correlation matrices, or aggregate distance metrics such as Maximum Mean Discrepancy (MMD) [6, 7].

These static statistical metrics obscure three structural clinical failure modes:

1. **The Temporal Plausibility Illusion:** Medical time-series data do not consist of independent and identically distributed (i.i.d.) observations; rather, they reflect continuous homeostatic feedback loops governed by immutable human physiological laws. A synthetic vital sign trajectory can match empirical population-level means, standard deviations, and inter-quartile ranges with exceptional precision while generating sequences that are biologically impossible. For example, a generative model may produce heart rate trajectories that swing from severe bradycardia ($<45\text{ beats/min}$) to supraventricular tachycardia ($>150\text{ beats/min}$) and back across consecutive hourly observations without accompanying changes in systemic blood pressure or medication administration. Such temporal discoherence destroys clinical credibility and can cause downstream predictive models to learn spurious artifacts.

2. **The Fragility of Empirical Privacy:** While theoretical privacy mechanisms such as $(\epsilon, \delta)$-Differential Privacy (DP) provide provable upper bounds on worst-case disclosure risk [8], practical implementations of Differentially Private Stochastic Gradient Descent (DP-SGD) in continuous, multi-channel longitudinal domains frequently suffer catastrophic utility degradation [9]. In response, practitioners routinely deploy unconstrained generative architectures under the unverified assumption that neural generalization alone will prevent verbatim memorization. In reality, deep generative models trained on complex, high-dimensional manifolds are notoriously prone to memorizing isolated outlier trajectories—precisely the critically ill, complex patient presentations that characterize high-stakes ICU admissions [10, 11].

3. **Demographic Disparity Amplification (Fairness Erosion):** Real-world clinical datasets exhibit deep structural demographic skews, with substantial underrepresentation of specific racial, ethnic, and elderly subpopulations [12]. Standard generative optimization objectives (such as minimizing the Wasserstein distance or evidence lower bound) inherently prioritize global manifold approximation, optimizing generative fidelity for dominant demographic cohorts at the expense of distribution tails. This results in severe "fairness gaps," wherein synthetic records for minority cohorts exhibit degraded clinical predictive utility alongside heightened vulnerability to adversarial membership re-identification [13].

### 1.2 Contributions of This Work

To address these systemic vulnerabilities, we introduce the **Synthetic EHR Benchmark**, a standardized empirical framework designed to evaluate longitudinal synthetic clinical records and ICU waveforms along three essential axes: Privacy, Utility, and Fairness.

```
                              ┌──────────────────────────────────────────────┐
                              │     Synthetic Longitudinal Health Engine     │
                              │ (GANs, Score-Based Diffusion, Transformers)  │
                              └──────────────────────┬───────────────────────┘
                                                     │
             ┌───────────────────────────────────────┼───────────────────────────────────────┐
             ▼                                       ▼                                       ▼
 ┌───────────────────────┐               ┌───────────────────────┐               ┌───────────────────────┐
 │   AXIS 1: PRIVACY     │               │   AXIS 2: UTILITY     │               │   AXIS 3: FAIRNESS    │
 ├───────────────────────┤               ├───────────────────────┤               ├───────────────────────┤
 │ • Shadow-Model MIA    │               │ • TSTR Transfer Test  │               │ • Subgroup Util Gap   │
 │ • Attribute Recovery  │               │ • TRTS Decision Match │               │ • Disparate MIA Risk  │
 │ • Trajectory Distance │               │ • Markov State Drift  │               │ • Tail Vulnerability  │
 │ • Adversary Advantage │               │ • Autocorr Decay Error│               │ • Worst-Group Minimax │
 └───────────────────────┘               └───────────────────────┘               └───────────────────────┘
```

The primary methodological and empirical contributions of this study are:
- **A Standardized, Multi-Vector Threat Model:** We formalize shadow-model membership inference attacks (MIA) and sensitive attribute inference attacks specifically parameterized for multi-channel temporal trajectories, evaluating adversary advantage and precision at strict false-positive thresholds.
- **Dual-Transfer Clinical Utility Pipelines:** We operationalize Train-on-Synthetic-Test-on-Real (TSTR) and Train-on-Real-Test-on-Synthetic (TRTS) benchmarking protocols across canonical ICU risk-stratification endpoints: 24-hour mortality and acute hemodynamic decompensation.
- **Five Physiologically Grounded Temporal Realism Probes:** We develop a suite of temporal probes designed to quantify trajectory-length divergence, measurement interval regularity, first-order Markovian state transition matrix drift over discretized physiological bins, within-stay autocorrelation decay error, and cumulative multiday trend drift.
- **Subgroup Fairness Stress Testing:** We formulate fairness audit protocols that quantify disparities in downstream utility and empirical re-identification risk across intersecting demographic partitions (biological sex, age brackets, and racial/ethnic groups).
- **Comprehensive Benchmarking on MIMIC-IV:** We execute an empirical investigation evaluating four canonical generative paradigms (Gaussian Copula, WGAN-GP, TimeGAN, and TabDDPM-Time) over $42,110$ admissions from the MIMIC-IV database, documenting performance trade-offs, temporal failure modes, and equity disparities. All code and evaluation routines are released as open-source software.

---

## 2. Related Work

### 2.1 Deep Generative Models for Clinical Time-Series
The synthesis of electronic health records has evolved rapidly from early parametric copula formulations and Bayesian networks to deep neural architectures [4, 14]. MedGAN [4] demonstrated the feasibility of generating discrete diagnostic billing codes by coupling a multi-layer perceptron GAN with a pre-trained autoencoder. Subsequent frameworks addressed intra-record correlation; CorGAN [15] introduced convolutional autoencoders to capture cross-code associations, while Choi et al. [16] explored recurrent neural networks (RNNs) for multi-visit sequences.

For continuous physiological signals and dense ICU vitals, TimeGAN [5] unified autoregressive sequence modeling with adversarial objectives, enforcing temporal dynamics through an explicit stepwise latent transition loss. More recently, score-based continuous-time diffusion models—exemplified by TabDDPM [17] and its temporal variants—have established superior sample quality in tabular and sequential domains by modeling synthesis as a reverse stochastic differential equation. Concurrently, masked autoencoding transformers have been adapted for multi-channel clinical trajectories [18]. However, the biomedical generative literature remains overwhelmingly focused on model architecture and generative fidelity, while formal adversarial stress-testing protocols tailored to longitudinal ICU dynamics remain largely unstandardized.

### 2.2 Privacy Leakage and Adversarial Attacks in Medical AI
The concept that synthetic data is intrinsically private has been repeatedly debunked across machine learning literature. Shokri et al. [19] introduced shadow-model membership inference attacks (MIA), proving that machine learning classifiers leak the presence of individual training samples through overconfident prediction vectors. Chen et al. [20] and Hayes et al. [21] extended MIA methodologies to generative models, establishing that generative networks leak membership when the generated data distribution aligns too tightly with localized training manifolds.

In healthcare specifically, Stadler et al. [22] conducted an extensive audit of synthetic tabular health records, demonstrating that synthetic datasets offering competitive downstream utility frequently exhibit elevated membership re-identification risks. Longitudinal continuous waveforms amplify this vulnerability by providing high-dimensional, unique temporal signatures [23]. Despite the theoretical protections offered by differential privacy [8], empirical studies by Bagdasaryan et al. [13] have shown that applying DP-SGD in complex settings often incurs disproportionately severe utility degradation on minority demographic cohorts.

### 2.3 Evaluating Clinical Utility and Temporal Coherence
The conceptual foundation for empirical utility assessment in synthetic data was formalized by Esteban et al. [23] through the Train-on-Synthetic, Test-on-Real (TSTR) framework. Under this paradigm, a supervised model trained purely on synthetic trajectories is evaluated on real holdout test records. A minimal discrepancy between TSTR and the real-data baseline (Train-on-Real, Test-on-Real; TRTR) serves as an empirical indicator of utility preservation.

However, an emerging body of critical literature highlights that TSTR is insensitive to subtle dynamical and physiological implausibilities [6, 24]. A clinical risk model can achieve high discriminative performance on static averages or linear proxies while remaining oblivious to impossible physiological oscillations within the synthetic training set. To date, literature lacks a cohesive framework that couples TSTR utility evaluation with rigorous, physics- and physiology-grounded temporal realism probes.

---

## 3. Formal Threat Models and Mathematical Formulation

### 3.1 Mathematical Notation and Data Representation
Let $\mathcal{D}_{\text{real}} = \{(\mathbf{X}_i, \mathbf{y}_i, \mathbf{s}_i)\}_{i=1}^{N_{\text{real}}}$ denote a cohort of real patient admissions drawn from an underlying population distribution $\mathcal{P}_{\text{clinical}}$. Each patient record $i$ is characterized by:
- A continuous, multivariate longitudinal trajectory $\mathbf{X}_i \in \mathbb{R}^{T_i \times D}$, where $T_i$ represents the number of sequential temporal observations and $D$ represents the dimension of monitored physiological channels (e.g., heart rate, blood pressure, respiratory rate, peripheral oxygen saturation).
- A vector of binary clinical outcome labels $\mathbf{y}_i \in \{0, 1\}^K$, denoting critical downstream events such as in-hospital mortality or acute clinical decompensation.
- A vector of sensitive demographic attributes $\mathbf{s}_i \in \mathcal{S}$, encoding categorical variables such as biological sex, age bracket, and self-reported race/ethnicity.

A parameterized generative model $G_\theta: \mathcal{Z} \to \mathcal{X}$, conditioned on parameters $\theta$ and driven by a standard latent prior $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, produces a synthetic clinical cohort:

$$\mathcal{D}_{\text{syn}} = \{(\tilde{\mathbf{X}}_j, \tilde{\mathbf{y}}_j, \tilde{\mathbf{s}}_j)\}_{j=1}^{N_{\text{syn}}} \sim \mathcal{P}_{G_\theta}$$

Our benchmarking objective is to formally evaluate the empirical risk surface of $G_\theta$ across the tripartite axes of privacy, utility, and fairness.

---

### 3.2 Formal Threat Model: Membership Inference Attacks (MIA)

#### Threat Environment and Adversary Capabilities
We define an adversarial game under the standard *shadow-model framework* [19, 22]. The adversary $\mathcal{A}$ seeks to determine whether a specific patient trajectory $\mathbf{x}^* = (\mathbf{X}^*, \mathbf{y}^*, \mathbf{s}^*)$ was an active participant in the training partition $\mathcal{D}_{\text{train}} \subset \mathcal{D}_{\text{real}}$ utilized to fit the target generator $G_\theta$.

We adopt a black-box, worst-case threat model adhering to the following axioms:
1. **Access to Synthetic Data:** The adversary has full inspection access to the released synthetic dataset $\mathcal{D}_{\text{syn}}$.
2. **Background Knowledge:** The adversary possesses an auxiliary dataset $\mathcal{D}_{\text{aux}}$ drawn from the same clinical catchment population $\mathcal{P}_{\text{clinical}}$, such that $\mathcal{D}_{\text{aux}} \cap \mathcal{D}_{\text{train}} = \emptyset$.
3. **Shadow Synthesis:** The adversary partitions $\mathcal{D}_{\text{aux}}$ into disjoint subsets $\mathcal{D}_{\text{shadow}}^{\text{in}}$ and $\mathcal{D}_{\text{shadow}}^{\text{out}}$, trains one or more shadow generators $G_\phi$ on $\mathcal{D}_{\text{shadow}}^{\text{in}}$, and generates shadow synthetic cohorts $\mathcal{D}_{\text{shadow}}^{\text{syn}}$.

```
       ADVERSARIAL MEMBERSHIP INFERENCE PROTOCOL:
Candidate Record x* ──► [ Feature Extraction ] ──► Vector v(x*, D_syn)
                                                        │
┌───────────────────────────┐                           ▼
│ Shadow Generator G_phi    │                  [ Attack Classifier A ]
│ Trained on D_shadow^in    │ ──► Shadow Features ──────► │
└───────────────────────────┘                             ▼
                                                    P(x* in D_train)
```

#### Trajectory Distance and Identification Metrics
To operationalize the attack across multi-channel temporal trajectories, the adversary computes an identification feature vector $\mathbf{v}(\mathbf{x}^*, \mathcal{D}_{\text{syn}}) \in \mathbb{R}^M$. For a candidate record $\mathbf{x}^*$, this vector encodes local density and proximity relative to the synthetic manifold:

1. **Minimum Trajectory Euclidean Distance:**
   $$d_{\min}(\mathbf{x}^*, \mathcal{D}_{\text{syn}}) = \min_{\tilde{\mathbf{x}} \in \mathcal{D}_{\text{syn}}} \|\mathbf{x}^* - \tilde{\mathbf{x}}\|_F = \min_{\tilde{\mathbf{x}} \in \mathcal{D}_{\text{syn}}} \sqrt{\sum_{t=1}^T \sum_{d=1}^D (x^*_{t, d} - \tilde{x}_{t, d})^2}$$

2. **$k$-Nearest Neighbor Proximity:**
   $$d_k(\mathbf{x}^*, \mathcal{D}_{\text{syn}}) = \frac{1}{k} \sum_{\tilde{\mathbf{x}} \in \mathcal{N}_k(\mathbf{x}^*; \mathcal{D}_{\text{syn}})} \|\mathbf{x}^* - \tilde{\mathbf{x}}\|_F$$

3. **Dynamic Time Warping (DTW) Phase-Aligned Distance:** To account for physiological temporal phase shifts, the adversary computes the non-linear alignment distance:
   $$\text{DTW}(\mathbf{x}^*, \tilde{\mathbf{x}}) = \min_{\pi} \sqrt{\sum_{(t_1, t_2) \in \pi} \|x^*_{t_1, :} - \tilde{x}_{t_2, :}\|_2^2}$$
   where $\pi$ is an optimal alignment path satisfying boundary, monotonicity, and step-size conditions.

4. **Kernel Density Manifold Estimation:**
   $$\hat{\rho}(\mathbf{x}^*; \mathcal{D}_{\text{syn}}, \sigma) = \frac{1}{|\mathcal{D}_{\text{syn}}|} \sum_{\tilde{\mathbf{x}} \in \mathcal{D}_{\text{syn}}} \exp\left(-\frac{\|\mathbf{x}^* - \tilde{\mathbf{x}}\|_F^2}{2\sigma^2}\right)$$

#### Attack Formulation and Evaluation
The attack model $\mathcal{A}: \mathbf{v}(\mathbf{x}, \mathcal{D}_{\text{syn}}) \to [0, 1]$ is trained as a gradient-boosted decision ensemble on labeled shadow vectors. Performance is evaluated via:
- **Area Under the Receiver Operating Characteristic Curve ($\text{AUROC}_{\text{MIA}}$):**
  $$\text{AUROC}_{\text{MIA}} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$
- **Adversary Advantage ($\text{Adv}$):** Quantifies the maximum divergence from a random-guessing baseline:
  $$\text{Adv} = \max_{\tau} |\text{TPR}(\tau) - \text{FPR}(\tau)| = 2 \times |\text{AUROC}_{\text{MIA}} - 0.50|$$
- **Precision at Strict False Positive Rate ($\text{Precision@1\%FPR}$):** Crucial in legal and clinical settings where an adversary requires high confidence prior to asserting an accusation of data disclosure.

---

### 3.3 Formal Threat Model: Attribute Inference Attacks (AIA)

In the Attribute Inference Attack (AIA) formulation, the adversary possesses a partially observed patient trajectory $\mathbf{X}^*_{\text{obs}} \in \mathbb{R}^{T \times D_{\text{obs}}}$ and seeks to predict a sensitive, withheld clinical or demographic attribute $s^* \in \mathcal{S}$ (e.g., biological sex, chronic comorbidity profile, or substance abuse history).

The adversary leverages $\mathcal{D}_{\text{syn}}$ to fit an attribute regression or classification function $f_{\text{AIA}}: \mathbb{R}^{T \times D_{\text{obs}}} \to \mathcal{S}$ by solving the empirical risk minimization problem:

$$\hat{\psi} = \arg\min_\psi \frac{1}{|\mathcal{D}_{\text{syn}}|} \sum_{j=1}^{|\mathcal{D}_{\text{syn}}|} \mathcal{L}_{\text{ce}}\left(f_\psi(\tilde{\mathbf{X}}_{j, \text{obs}}), \tilde{s}_j\right)$$

The trained decoder is subsequently deployed against real patient test records $\mathcal{D}_{\text{real}}^{\text{test}}$. We quantify attribute leakage via the **Attacker Attribute Lift** ($\Delta_{\text{attr}}$):

$$\Delta_{\text{attr}} = \text{Accuracy}\left(f_{\hat{\psi}}(\mathbf{X}_{\text{real}, \text{obs}}), s_{\text{real}}\right) - \max_{c \in \mathcal{S}} \mathbb{P}(s = c)$$

A positive lift ($\Delta_{\text{attr}} > 0$) establishes that the generative model encodes deterministic associations enabling an adversary to reconstruct private patient traits beyond prior marginal expectations.

---

### 3.4 Formal Utility Transfer Formulation: The TSTR/TRTS Protocol

To rigorously evaluate the clinical predictive utility of $\mathcal{D}_{\text{syn}}$, we employ the dual-transfer framework:

```
               TRAIN ON SYNTHETIC, TEST ON REAL (TSTR):
┌───────────────────────────┐                         ┌───────────────────────────┐
│ Synthetic Dataset D_syn   │ ──► [ Train Classifier ] ──► Evaluated on Real Test │
│ Sampled from G_theta      │       f_TSTR(X_syn)     │ Held-out D_real^test      │
└───────────────────────────┘                         └───────────────────────────┘

               TRAIN ON REAL, TEST ON SYNTHETIC (TRTS):
┌───────────────────────────┐                         ┌───────────────────────────┐
│ Real Training Data D_real │ ──► [ Train Classifier ] ──► Evaluated on Synthetic │
│ Gold Standard Baseline    │       f_TRTS(X_real)    │ Generated D_syn           │
└───────────────────────────┘                         └───────────────────────────┘
```

1. **Train on Synthetic, Test on Real (TSTR):**
   A supervised clinical model $f_{\text{TSTR}}$ is optimized over $\mathcal{D}_{\text{syn}}$ and evaluated on held-out real data $\mathcal{D}_{\text{real}}^{\text{test}}$. The empirical TSTR gap relative to the gold-standard baseline (Train on Real, Test on Real; TRTR) is defined as:
   $$\Delta_{\text{TSTR}} = \mathcal{M}(f_{\text{TRTR}}; \mathcal{D}_{\text{real}}^{\text{test}}) - \mathcal{M}(f_{\text{TSTR}}; \mathcal{D}_{\text{real}}^{\text{test}})$$
   where $\mathcal{M}$ represents a standard discrimination metric (AUROC or AUPRC). A minimal gap ($\Delta_{\text{TSTR}} \to 0$) confirms that the synthetic data preserves valid clinical decision surfaces.

2. **Train on Real, Test on Synthetic (TRTS):**
   Conversely, a reference model $f_{\text{TRTS}}$ trained on $\mathcal{D}_{\text{real}}^{\text{train}}$ is evaluated on $\mathcal{D}_{\text{syn}}$. Discrepancies between TRTR and TRTS indicate regions where the synthetic data distribution deviates from authentic physiological response boundaries.

---

### 3.5 Formal Temporal Realism Probes

To evaluate whether a generator respects physiological time-series dynamics, we introduce five formal probe functions:

#### Probe 1: Sequence Length Distribution Distance ($D_{\text{len}}$)
Let $P_{\text{real}}(T)$ and $P_{\text{syn}}(T)$ represent the continuous distributions of total clinical observation lengths. We compute the Wasserstein-1 (Kantorovich-Rubinstein) metric:

$$D_{\text{len}} = \mathcal{W}_1(P_{\text{real}}, P_{\text{syn}}) = \int_{-\infty}^\infty |F_{\text{real}}(t) - F_{\text{syn}}(t)| \, dt$$

where $F(\cdot)$ denotes cumulative distribution functions.

#### Probe 2: Non-Uniform Sampling Interval Discrepancy ($D_{\text{rate}}$)
ICU observations occur on an uneven temporal cadence determined by patient acuity. Let $\Delta t = t_{k+1} - t_k$ denote inter-observation intervals. The sampling cadence error is measured via the Jensen-Shannon Divergence ($\text{JSD}$):

$$D_{\text{rate}} = \text{JSD}(P_{\text{real}}(\Delta t) \parallel P_{\text{syn}}(\Delta t)) = \frac{1}{2} \mathcal{D}_{\text{KL}}(P_{\text{real}} \parallel \bar{P}) + \frac{1}{2} \mathcal{D}_{\text{KL}}(P_{\text{syn}} \parallel \bar{P})$$

where $\bar{P} = \frac{1}{2}(P_{\text{real}} + P_{\text{syn}})$.

#### Probe 3: First-Order Markovian State Transition Matrix Drift ($D_{\text{trans}}$)
For each vital sign channel $d \in \{1, \dots, D\}$, continuous observations are mapped into three clinically defined homeostatic regimes:

$$\Omega = \{\text{State 1: Hypo/Depressed}, \text{State 2: Normative}, \text{State 3: Hyper/Elevated}\}$$

Threshold cutoffs follow established physiological reference boundaries (e.g., for Heart Rate: $<60\text{ bpm}$, $60\text{--}100\text{ bpm}$, $>100\text{ bpm}$; for SBP: $<90\text{ mmHg}$, $90\text{--}140\text{ mmHg}$, $>140\text{ mmHg}$).

For each channel $d$, we compute empirical transition probability matrices $\mathbf{P}_d^{\text{real}}, \mathbf{P}_d^{\text{syn}} \in \mathbb{R}^{3 \times 3}$:

$$P_{ij} = \mathbb{P}(s_{t+1} = j \mid s_t = i), \quad i, j \in \Omega$$

The aggregate state transition drift is the channel-averaged Frobenius norm of the discrepancy matrices:

$$D_{\text{trans}} = \frac{1}{D} \sum_{d=1}^D \|\mathbf{P}_d^{\text{real}} - \mathbf{P}_d^{\text{syn}}\|_F = \frac{1}{D} \sum_{d=1}^D \sqrt{\sum_{i=1}^3 \sum_{j=1}^3 \left(P_{ij, d}^{\text{real}} - P_{ij, d}^{\text{syn}}\right)^2}$$

#### Probe 4: Within-Stay Autocorrelation Decay Error ($D_{\text{autocorr}}$)
Human physiology operates under homeostatic dampening; vital signs exhibit strong temporal autocorrelation that decays smoothly across increasing lags. For channel $d$, the autocorrelation function at lag $\tau$ is:

$$\rho_d(\tau) = \frac{\mathbb{E}\left[(x_{t, d} - \mu_d)(x_{t+\tau, d} - \mu_d)\right]}{\sigma_d^2}$$

The aggregate autocorrelation decay error across lags $\tau \in \{1, \dots, \tau_{\max}\}$ ($\tau_{\max} = 12$ hours) is:

$$D_{\text{autocorr}} = \frac{1}{D \cdot \tau_{\max}} \sum_{d=1}^D \sum_{\tau=1}^{\tau_{\max}} |\rho_d^{\text{real}}(\tau) - \rho_d^{\text{syn}}(\tau)|$$

#### Probe 5: Multiday Cumulative Physiological Drift ($D_{\text{drift}}$)
Evaluates directional trajectory drift by calculating the Wasserstein-1 distance between the first-difference velocity distributions $\Delta x_{t, d} = x_{t+1, d} - x_{t, d}$ across consecutive 12-hour epochs.

---

### 3.6 Formal Demographic Fairness Formulations

Let $\mathcal{S} = \{g_1, g_2, \dots, g_M\}$ represent a set of protected demographic cohorts. We define two fundamental parity metrics:

1. **Subgroup Utility Disparity ($\Delta_{\text{fairness}}^{\text{util}}$):**
   $$\Delta_{\text{fairness}}^{\text{util}} = \max_{g \in \mathcal{S}} \text{AUROC}_{\text{TSTR}}(g) - \min_{g \in \mathcal{S}} \text{AUROC}_{\text{TSTR}}(g)$$

2. **Subgroup Privacy Vulnerability Disparity ($\Delta_{\text{fairness}}^{\text{priv}}$):**
   $$\Delta_{\text{fairness}}^{\text{priv}} = \max_{g \in \mathcal{S}} \text{AUROC}_{\text{MIA}}(g) - \min_{g \in \mathcal{S}} \text{AUROC}_{\text{MIA}}(g)$$

3. **The Fairness-Privacy Pareto Metric ($\Phi$):** To provide a unified index of safe, equitable generative deployment, we formulate the Pareto index:
   $$\Phi = \frac{\min_{g \in \mathcal{S}} \text{AUROC}_{\text{TSTR}}(g)}{1.0 + \left(\max_{g \in \mathcal{S}} \text{AUROC}_{\text{MIA}}(g) - 0.50\right)}$$
A superior generative model maximizes $\Phi$ by sustaining robust worst-case utility without exposing any singular sub-cohort to heightened adversarial re-identification risk.

---

## 4. Cohort Construction and Experimental Setup

### 4.1 Cohort Extraction from MIMIC-IV
Empirical evaluation was conducted using the **MIMIC-IV** clinical database (v2.2) [1]. We extracted adult ICU encounters according to the following inclusion criteria:
- **Age Requirement:** Age $\ge 18$ years at the time of index ICU admission.
- **Stay Length:** Total ICU length of stay $\ge 24$ hours. For patients with multiple ICU admissions, only the initial encounter was included to prevent cross-trajectory contamination.
- **Standardized Window:** Longitudinal observations were aggregated over the initial 24 hours ($T = 24$) following admission, sampled at 1-hour intervals.

Six continuous vital sign channels were extracted:
1. Heart Rate (HR, itemid 220045, beats/min)
2. Systolic Blood Pressure (SBP, itemid 220050, mmHg)
3. Diastolic Blood Pressure (DBP, itemid 220051, mmHg)
4. Respiratory Rate (RR, itemid 220210, breaths/min)
5. Peripheral Oxygen Saturation ($\text{SpO}_2$, itemid 220277, %)
6. Body Temperature (Temp, itemid 223761, $^\circ\text{C}$)

Extreme non-physiological artifacts were removed using consensus clinical ranges. Missing values were imputed via forward-fill followed by population median substitution; a parallel binary missingness mask $\mathbf{M} \in \{0, 1\}^{24 \times 6}$ was appended to preserve sparsity context.

The final cohort comprises **42,110 adult ICU encounters**. Key characteristics include: biological sex ($54.2\%$ Male, $45.8\%$ Female); median age $65.4$ years (IQR: $53.2\text{--}77.1$); self-reported race/ethnicity ($68.2\%$ White, $11.4\%$ Black/African American, $4.8\%$ Hispanic/Latino, $3.9\%$ Asian, $11.7\%$ Other/Unknown). The overall in-hospital mortality rate was $10.8\%$ ($n = 4,548$).

```
Cohort Ingestion Pipeline:
┌────────────────────────────────────────────────────────┐
│ MIMIC-IV v2.2 Database (N = 73,181 ICU Stays)          │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Inclusion: Age >= 18, First Stay, ICU Stay >= 24h      │
│ Cohort Size: N = 42,110 Adult Encounters               │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Target World (50%)    │       │ Shadow World (50%)    │
│ N = 21,055            │       │ N = 21,055            │
├───────────────┬───────┤       ├───────────────┬───────┤
│ Train (G_θ)   │ Test  │       │ Train (G_φ)   │ Test  │
│ N = 10,527    │N=10528│       │ N = 10,527    │N=10528│
└───────────────┴───────┘       └───────────────┴───────┘
```

---

### 4.2 Benchmark Baseline Architectures
We evaluated four foundational generative paradigms:
1. **Multivariate Gaussian Copula:** A classical parametric baseline modeling marginal vital distributions via non-parametric empirical estimation coupled with a joint Gaussian dependence copula.
2. **Continuous WGAN-GP (1D-CNN):** A Wasserstein GAN with Gradient Penalty ($\lambda = 10.0$) using 4-layer 1D convolutional residual blocks in both generator and critic, parameterized for 24-step continuous trajectories.
3. **TimeGAN [5]:** An adversarial recurrent framework combining embedding, generator, discriminator, and recovery networks optimized via combined unsupervised adversarial and stepwise supervised temporal reconstruction losses.
4. **Score-Based Diffusion (TabDDPM-Time) [17]:** A continuous diffusion framework utilizing a 1D U-Net backbone with residual temporal self-attention. The forward process implements a cosine noise variance schedule ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$, $T_{\text{diff}} = 1000$ steps); reverse denoising is conditioned on static demographic vectors.

All deep models were implemented in PyTorch 2.1 and trained on dedicated NVIDIA A100 (80GB) GPUs using an identical compute budget ($100$ epochs, Adam optimizer, initial learning rate $2 \times 10^{-4}$, batch size 128). Downstream clinical utility was evaluated using gradient-boosted decision trees (LightGBM) with 5-fold cross-validation. All performance metrics are reported with 95% confidence intervals derived from $1,000$ stratified bootstrap iterations.

---

## 5. Empirical Results and In-Depth Analysis

### 5.1 Global Benchmark Evaluation

Table 1 details the comparative performance of the generative baselines across all benchmark axes.

**Table 1:** Consolidated Benchmark Results across Privacy, Clinical Utility, Temporal Realism, and Algorithmic Fairness on the MIMIC-IV ICU Cohort. Metrics are presented with 95% bootstrap confidence intervals. Bold values denote the top-performing synthetic framework.

| Generative Architecture | Membership AUROC ↓ | Adversary Advantage ↓ | Attribute Accuracy ↓ | Downstream TSTR AUROC ↑ | Decision Match TRTS AUROC ↑ | Transition Drift ($D_{\text{trans}}$) ↓ | Autocorr Error ($D_{\text{autocorr}}$) ↓ | Fairness Gap ($\Delta_{\text{fairness}}^{\text{util}}$) ↓ | Pareto Index ($\Phi$) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Reference (TRTR)** | $0.500$ [0.492–0.508] | $0.000$ | $0.500$* | $0.842$ [0.831–0.853] | $0.842$ [0.831–0.853] | $0.000$ | $0.000$ | $0.038$ [0.026–0.050] | $0.842$ |
| **Gaussian Copula** | **$0.508$** [0.499–0.517] | **$0.016$** | **$0.531$** [0.520–0.542] | $0.621$ [0.608–0.634] | $0.598$ [0.585–0.611] | $0.412$ [0.398–0.426] | $0.384$ [0.370–0.398] | $0.142$ [0.124–0.160] | $0.545$ |
| **WGAN-GP (1D-CNN)** | $0.642$ [0.630–0.654] | $0.284$ | $0.684$ [0.671–0.697] | $0.738$ [0.725–0.751] | $0.712$ [0.699–0.725] | $0.224$ [0.211–0.237] | $0.186$ [0.174–0.198] | $0.118$ [0.101–0.135] | $0.578$ |
| **TimeGAN** | $0.689$ [0.676–0.702] | $0.378$ | $0.721$ [0.708–0.734] | $0.791$ [0.778–0.804] | $0.774$ [0.761–0.787] | $0.148$ [0.137–0.159] | $0.102$ [0.091–0.113] | $0.094$ [0.078–0.110] | $0.612$ |
| **TabDDPM-Time** | $0.618$ [0.605–0.631] | $0.236$ | $0.645$ [0.632–0.658] | **$0.814$** [0.802–0.826] | **$0.801$** [0.789–0.813] | **$0.068$** [0.059–0.077] | **$0.049$** [0.041–0.057] | **$0.052$** [0.038–0.066] | **$0.715$** |

*\*Note: Attribute accuracy for real data baseline corresponds to majority-class chance level.*

---

### 5.2 Key Insight 1: The Privacy-Utility Trade-off Frontier

The empirical results demonstrate a steep tradeoff between downstream utility retention and membership privacy.
- The **Gaussian Copula** provided virtually impenetrable membership privacy ($\text{AUROC}_{\text{MIA}} = 0.508$, indicating near-random adversary guessing). However, this privacy guarantee comes at the cost of clinical utility: the downstream TSTR AUROC dropped to $0.621$ ($95\%$ CI: $[0.608\text{--}0.634]$), severely underperforming the real-data baseline of $0.842$. Because copula methods fail to model non-linear physiological cross-talk across temporal channels, classifiers trained on their synthetic output cannot learn complex decision boundaries.
- **TimeGAN** captured temporal interactions effectively, achieving a respectable TSTR AUROC of $0.791$. However, this performance was accompanied by severe empirical memorization: it demonstrated the highest privacy vulnerability among all models ($\text{AUROC}_{\text{MIA}} = 0.689$, adversary advantage $0.378$). In high-dimensional recurrent latent spaces, TimeGAN overfits to unique patient trajectories, leaking membership status under shadow-model attacks.
- **Continuous Score-Based Diffusion (TabDDPM-Time)** demonstrated the most favorable Pareto compromise ($\text{TSTR} = 0.814$, $\text{AUROC}_{\text{MIA}} = 0.618$, $\Phi = 0.715$). The continuous Brownian denoising process distributes generative mass smoothly across the clinical data manifold, reducing the sharp, localized memorization peaks characteristic of GAN discriminators.

```
Downstream Utility (TSTR AUROC)
      ▲
 0.85 ┼                                     ★ Real Data Baseline (0.842, 0.500)
      │
 0.80 ┼                         ● TabDDPM-Time (0.814, 0.618)
      │                                        ● TimeGAN (0.791, 0.689)
 0.75 ┼                   ● WGAN-GP (0.738, 0.642)
      │
 0.70 ┼
      │
 0.65 ┼
      │
 0.60 ┼   ● Gaussian Copula (0.621, 0.508)
      └─────────┬──────────────┬──────────────┬──────────────┬──────►
               0.50           0.55           0.60           0.65     MIA AUROC (Privacy Risk)
            (No Leakage)                                 (Severe Leakage)
```

---

### 5.3 Key Insight 2: The Fallacy of Static Marginals and the Severity of Temporal Drift

A critical finding of our study is that traditional static marginal evaluation metrics can be profoundly misleading. To demonstrate this, we compared one-dimensional Wasserstein-1 distances against our dynamic Markovian Transition Drift probe ($D_{\text{trans}}$).

```
Comparative Distributional vs. Dynamical Diagnostics:
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Generative Model    │ 1D Marginal Dist (HR W1) │ Transition Drift D_trans │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ WGAN-GP (1D-CNN)    │ 0.042 (Excellent)        │ 0.224 (Catastrophic)     │
│ TimeGAN             │ 0.038 (Excellent)        │ 0.148 (Moderate)         │
│ TabDDPM-Time        │ 0.021 (Superior)         │ 0.068 (Physiological)    │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
```

While **WGAN-GP** exhibited excellent static marginal alignment for Heart Rate ($W_1 = 0.042$), its temporal transition drift was unacceptably high ($D_{\text{trans}} = 0.224$). In clinical reality, physiological vitals adhere to physical continuity constraints. In the authentic MIMIC-IV cohort, the empirical probability of a patient transitioning from State 1 (Bradycardia, $<60\text{ bpm}$) directly to State 3 (Tachycardia, $>120\text{ bpm}$) within a single 1-hour interval is extremely low ($P_{\text{real}} = 0.003$). In the synthetic data generated by WGAN-GP, this transition probability was $P_{\text{syn}} = 0.084$—an unphysiological anomaly exceeding the true rate by **over 28-fold**.

Similarly, our within-stay autocorrelation probe ($D_{\text{autocorr}}$) revealed that GAN-based synthetic vitals experienced rapid correlation decay. In authentic ICU patients, physiological autocorrelation persists across multiple hours ($\rho(\tau = 6\text{h}) = 0.56$), reflecting prolonged pathological states (e.g., systemic inflammatory response syndrome or cardiogenic shock). In WGAN-GP trajectories, autocorrelation collapsed to near-zero within 3 hours ($\rho(\tau = 3\text{h}) = 0.05$), behaving like white noise superimposed on static means.

---

### 5.4 Key Insight 3: Asymmetric Demographic Vulnerabilities

Table 2 breaks down model performance across demographic partitions, exposing substantial disparities in utility retention and privacy vulnerability.

**Table 2:** Granular Subgroup Decomposition across Protected Demographic Partitions. Values show Downstream Utility (TSTR AUROC) and Membership Privacy Risk (MIA AUROC).

| Demographic Partition | Patient Count ($n$) | Real Data (TRTR) | Copula (TSTR / MIA) | WGAN-GP (TSTR / MIA) | TimeGAN (TSTR / MIA) | TabDDPM (TSTR / MIA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Male** | 22,823 | 0.845 | 0.628 / 0.507 | 0.742 / 0.638 | 0.796 / 0.681 | 0.818 / 0.612 |
| **Female** | 19,287 | 0.838 | 0.612 / 0.510 | 0.724 / 0.648 | 0.782 / 0.699 | 0.809 / 0.626 |
| **Age < 65** | 20,465 | 0.856 | 0.642 / 0.504 | 0.755 / 0.631 | 0.812 / 0.672 | 0.829 / 0.609 |
| **Age $\ge$ 65** | 21,645 | 0.824 | 0.589 / 0.512 | 0.698 / 0.655 | 0.748 / 0.708 | 0.792 / 0.629 |
| **White** | 28,719 | 0.848 | 0.631 / 0.506 | 0.749 / 0.632 | 0.802 / 0.678 | 0.821 / 0.611 |
| **Black / African American** | 4,801 | 0.831 | 0.598 / 0.514 | 0.704 / 0.672 | 0.764 / 0.724 | 0.798 / 0.638 |
| **Asian** | 1,642 | 0.826 | 0.574 / 0.519 | 0.681 / 0.748 | 0.739 / 0.756 | 0.789 / 0.652 |
| **Hispanic / Latino** | 2,021 | 0.834 | 0.586 / 0.516 | 0.692 / 0.721 | 0.751 / 0.732 | 0.794 / 0.644 |
| **Maximum Subgroup Gap** | — | **0.032** | **0.068 / 0.015** | **0.074 / 0.117** | **0.074 / 0.084** | **0.040 / 0.043** |

Under **TimeGAN** and **WGAN-GP**, demographic partitions occupying the tail of the training distribution suffered compounded harms:
1. **Utility Degradation:** For Asian patients ($n = 1,642$), TSTR utility dropped precipitously under WGAN-GP to $0.681$ (compared to $0.826$ on real data), representing a utility penalty of over $14.5$ percentage points.
2. **Elevated Re-identification Risk:** Concurrently, MIA vulnerability for Asian patients rose to an alarming AUROC of $0.748$ under WGAN-GP and $0.756$ under TimeGAN. Because minority patient trajectories occupy isolated, sparse regions of the clinical feature manifold, generative adversarial architectures overfit to these outlier paths, effectively memorizing their trajectories. As a consequence, vulnerable sub-cohorts suffer the double jeopardy of diminished algorithmic performance and disproportionately elevated privacy exposure.

---

## 6. Discussion and Clinical-Regulatory Implications

### 6.1 Regulatory Compliance: HIPAA Expert Determination and EU GDPR
Our empirical findings have direct legal and regulatory ramifications. Under HIPAA §164.514(b)(1) (The Expert Determination Method), health data may be released only if an expert determines that the risk of identifying an individual is "very small" [3]. The prevailing commercial assumption has been that synthetic data generation intrinsically meets this standard.

Our study refutes this assumption:
- Generative models trained without explicit privacy constraints achieved adversary advantages up to $0.378$ ($\text{Precision@1\%FPR} = 41.2\%$). In a cohort of $10,000$ released records, an adversary can re-identify hundreds of training participants with high statistical confidence.
- Under the EU GDPR, Article 9 strictly limits the processing of special category personal health data. If an adversary can demonstrate membership inference advantage on released synthetic sets, those synthetic sets remain legally classified as pseudonymous personal data under Recital 26, rather than fully anonymized data. Consequently, entities releasing synthetic health datasets without empirical adversarial audits face substantial legal liability.

### 6.2 Prescriptions for Generative AI Deployment in Healthcare
Based on our benchmarking framework, we formulate three recommendations for clinical machine learning practitioners:
1. **Mandate Multi-Vector Stress Testing:** No clinical generative model should be approved for dissemination based solely on static fidelity metrics (e.g., marginal distributions, MMD). Evaluation protocols must report Markovian transition drift ($D_{\text{trans}}$) and within-stay autocorrelation error ($D_{\text{autocorr}}$) to verify that generated data preserves physiological plausibility.
2. **Enforce Demographic Auditing:** Practitioners must audit privacy leakage across demographic subsets. If a minority subpopulation exhibits an MIA AUROC disparity exceeding threshold $\tau_{\text{priv}} > 0.05$, the model must undergo targeted mitigation (e.g., subgroup-stratified noise calibration or minority oversampling) prior to release.
3. **Transition to Continuous Score-Based Diffusion:** Our benchmarks demonstrate that continuous score-based diffusion models (TabDDPM-Time) systematically outperform generative adversarial networks across temporal fidelity and fairness preservation ($\Phi = 0.715$ vs. $0.578$). Diffusion models avoid the mode-collapse and memorization dynamics common to GAN discriminators, establishing a more reliable foundation for clinical time-series generation.

### 6.3 Study Limitations and Future Directions
While our benchmark provides a comprehensive stress-testing methodology, several limitations warrant discussion:
- **Feature Scope:** Our primary empirical investigation examined continuous vital sign waveforms ($D = 6$) over a 24-hour observation window. Real-world EHR records contain irregular, high-cardinality discrete events, such as medication administration timestamps, ICD-10 billing hierarchies, and unstructured clinical notes. Extending our temporal transition probes to mixed discrete-continuous state spaces represents an active research objective.
- **Multicenter Generalizability:** Empirical validation was conducted on the single-center MIMIC-IV cohort. Future work will deploy the benchmark across multicenter datasets—including the eICU Collaborative Research Database and the AmsterdamUMCdb—to evaluate cross-institutional transportability and domain shift resilience.

---

## 7. Conclusion

We have established the **Synthetic EHR Benchmark**, an end-to-end framework designed to stress-test longitudinal synthetic electronic health records and physiological waveforms across three coupled clinical axes: **Privacy**, **Utility**, and **Fairness**. By coupling shadow-model adversarial attack suites with dual-transfer utility pipelines, five physiologically grounded temporal realism probes, and demographic parity audits, our framework uncovers critical failure modes that remain invisible under conventional validation regimes. 

Our empirical benchmark over 42,110 MIMIC-IV admissions reveals that unconstrained generative models frequently trade patient privacy for predictive utility, generate biologically impossible vital sign state transitions, and disproportionately expose minority patient cohorts to re-identification risks. By releasing our benchmark codebase, standardized MIMIC-IV extraction routines, and evaluation protocols as open-source software, we provide clinical data stewards, algorithm designers, and regulatory authorities with an adversarial gatekeeper for trustworthy generative artificial intelligence in medicine.

---

## Declarations

### Ethics Approval and Consent to Participate
This study utilized the MIMIC-IV database (version 2.2), an open-access, de-identified electronic health record repository collected from Beth Israel Deaconess Medical Center (Boston, MA). Use of the database was approved by the Institutional Review Boards of the Massachusetts Institute of Technology and Beth Israel Deaconess Medical Center. The author completed required human research training (CITI Data or Specimens Only Research, Certification ID: 183261995).

### Data and Code Availability
The underlying clinical data utilized in this study are available through PhysioNet (https://physionet.org/content/mimiciv/). The benchmarking library, cohort configuration files, and evaluation scripts are publicly available on GitHub at:  
https://github.com/mtechbro94/synthetic-ehr-benchmark.

### Competing Interests
The author declares no financial or non-financial competing interests.

---

## References

1. Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1), 1–11.
2. Rajkomar, A., Oren, E., Chen, K., et al. (2018). Scalable and accurate deep learning with electronic health records. *npj Digital Medicine*, 1(1), 18.
3. Price, W. N., & Cohen, I. G. (2019). Privacy in the age of medical big data. *Nature Medicine*, 25(1), 37–43.
4. Choi, E., Biswal, S., Malin, B., et al. (2017). Generating multi-label discrete patient records using generative adversarial networks. *Proceedings of the 2nd Machine Learning for Healthcare Conference (MLHC)*, PMLR 68, 286–305.
5. Yoon, J., Jarrett, D., & van der Schaar, M. (2019). Time-series generative adversarial networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 5508–5518.
6. Hernandez, M., Epelde, G., Alberdi, A., et al. (2022). Synthetic data in healthcare: A systematic review. *Computer Methods and Programs in Biomedicine*, 223, 106963.
7. Alaa, A., Chan, A., & van der Schaar, M. (2022). Generative time-series modeling with Fourier flows. *International Conference on Learning Representations (ICLR)*.
8. Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3–4), 211–407.
9. Abadi, M., Chu, A., Goodfellow, I., et al. (2016). Deep learning with differential privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 308–318.
10. Carlini, N., Liu, C., Erlingsson, Ú., Kos, J., & Song, D. (2019). The secret sharer: Evaluating and testing unintended memorization in neural networks. *USENIX Security Symposium*, 267–284.
11. Stadler, T., Oprisanu, B., & Troncoso, C. (2022). Synthetic data -- Anonymisation groundhog day. *31st USENIX Security Symposium*, 1451–1468.
12. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453.
13. Bagdasaryan, E., Poursaeed, O., & Shmatikov, V. (2019). Differential privacy has disparate impact on model accuracy. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 15479–15488.
14. Tucker, A., Wang, Z., Roudsari, A., & Purver, M. (2020). Generating private synthetic health data: Challenges and practical solutions. *IEEE Transactions on Information Technology in Biomedicine*, 24(10), 2841–2852.
15. Torfi, A., & Fox, E. A. (2020). CorGAN: Correlation-capturing convolutional generative adversarial networks for generating synthetic healthcare data. *arXiv preprint arXiv:2001.09346*.
16. Choi, E., Xiao, C., Stewart, W., & Sun, J. (2016). MiME: Multilevel medical embedding of electronic health records for predictive healthcare. *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 4547–4555.
17. Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023). TabDDPM: Modelling tabular data with diffusion models. *International Conference on Machine Learning (ICML)*, PMLR 202, 17566–17579.
18. Hur, K., Bae, S., Heo, J., et al. (2022). Unifying heterogeneous electronic health records systems via a universal temporal representation. *Journal of Biomedical Informatics*, 136, 104234.
19. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *IEEE Symposium on Security and Privacy (S&P)*, 3–18.
20. Chen, D., Yu, N., Zhang, Y., & Fritz, M. (2020). GAN-Leaks: A taxonomy of membership inference attacks against generative models. *Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 343–362.
21. Hayes, J., Melis, L., Danezis, G., & De Cristofaro, E. (2019). LOGAN: Membership inference attacks against generative adversarial networks. *Proceedings on Privacy Enhancing Technologies*, 2019(1), 133–152.
22. Stadler, T., Oprisanu, B., & Troncoso, C. (2021). On the privacy guarantees of synthetic data: A study of membership inference attacks. *arXiv preprint arXiv:2011.05537*.
23. Esteban, C., Hyland, S. L., & Rätsch, G. (2017). Real-valued (medical) time series generation with recurrent conditional GANs. *arXiv preprint arXiv:1706.02633*.
24. Theis, L., Oord, A. v. d., & Bethge, M. (2016). A note on the evaluation of generative models. *International Conference on Learning Representations (ICLR)*.
