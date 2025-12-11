START
│
├─► 1. Nail down the causal target (ATT of what population? when?)            ──[Guide Step 1]:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
│
├─► 2. What does your panel look like?
│       │
│       ├─ Exactly 2 periods & 2 groups → Classic 2×2 DiD  
│       │     Estimator = ΔYtreated − ΔYcontrol.  Stop here.
│       │
│       └─ More periods / staggered entry → go to 3
│
├─► 3. Pick a comparison group assumption
│       │
│       ├─ **Never-treated units available & credible?**  
│       │        Yes → Assumption PT-GT-Nev → Estimator:  
│       │              Callaway-Sant’Anna “never” or Sun-Abraham cohort-time averages:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
│       │
│       └─ No → Use not-yet-treated units  
│                Assumption PT-GT-NYT → Estimator:  
│                Callaway-Sant’Anna “nyt”, stacked DiD, local projections:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
│
├─► 4. Will you impose *full* parallel trends across **all** groups & times?  
│       │           (PT-GT-all — strong and risky)  
│       ├─ Yes → Estimator options: Extended TWFE (Wooldridge 2021),  
│       │        Borusyak-Jaravel-Spiess, de Chaisemartin-D’Haultfoeuille:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}  
│       └─ No  → stick with comparison group choice from step 3
│
├─► 5. Are treatment effects plausibly constant across cohorts/time?  
│       │
│       ├─ Yes → You *may* report two-way fixed-effects (TWFE)  
│       │        but first decompose the weights (e.g., `bacondecomp`)  
│       │        to check for negative or perverse weighting.  
│       │
│       └─ No  → Drop TWFE; rely on block-based estimators.  
│                 (TWFE mixes already-treated controls and can flip signs.):contentReference[oaicite:8]{index=8}
│
├─► 6. Covariate imbalance between treated and controls?
│       │
│       ├─ No → stay unconditional.  
│       │
│       └─ Yes → Impose **Conditional Parallel Trends** and choose:  
│                • Regression Adjustment (RA)  
│                • Inverse Probability Weighting (IPW)  
│                • Doubly-Robust (DR) combo  
│                (Sant’Anna-Zhao 2020; Callaway-Sant’Anna 2021):contentReference[oaicite:9]{index=9}
│
├─► 7. Decide the weighting scheme ω up-front  
│       (unit-level ATT vs person-level ATT; weights shape the estimand):contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
│
├─► 8. Inference & robustness  
│       • Cluster appropriately; discuss what’s random (Guide Step 4):contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}  
│       • Plot pre-trends for every comparison block.  
│       • Run sensitivity/bounds if pre-trends look shaky (Rambachan-Roth etc.).
│
└─► 9. If any assumption above fails → abandon DiD or redesign.  
        (Forward-engineer, don’t reverse-engineer.):contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}
END
