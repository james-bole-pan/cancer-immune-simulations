# **Cancer–Immune Simulations**
**MIT 6.7300 — Advances in Computational Modeling**  
Instructor: **Dr. Luca Daniel**  
Authors: **James Bole Pan, Alex Hochroth, Max Misterka**


## **Overview**
This project implements a reaction–diffusion tumor–immune model to evaluate whether pretreatment spatial biopsy data contain predictive structure for immunotherapy response. We benchmark explicit and implicit numerical solvers, use SGD-based parameter calibration, and assess model performance on real pretreatment spatial biopsy initial conditions from Chen et al., *Nature Immunology* (2024): https://www.nature.com/articles/s41590-024-01792-2.

The calibrated model achieves 69.6% accuracy in distinguishing responders from non-responders using **pretreatment spatial tumor–immune topology alone**, demonstrating that spatial immune organization could carry meaningful predictive signal and motivating future modeling approaches that include more mechanistic details in model design.

## **Project Materials**
- **Project Report:** https://drive.google.com/file/d/1m3jh0qpKOa93byfTeFYhBIweQHVyacrz/view?usp=drive_link  
- **Project Slides:** https://drive.google.com/file/d/1g0AyGTduYVQeOW_u8O0eKnMG6lmOW7Av/view?usp=sharing
  

## **Example Simulation**
Example of a clinically responsive (R) sample.  
- **No-drug simulation:** uncontrolled tumor expansion, minimal immune activation  
- **With-drug simulation:** strong T-cell recruitment and tumor regression  

**No Drug:**  
![no_drug](https://github.com/user-attachments/assets/55d66b9d-ed11-42af-9e8f-d89d901e19be)

**With Drug:**  
![with_drug](https://github.com/user-attachments/assets/897645f8-e9c1-410b-8bd3-d19164d5c9c6)

**Evolutionary Trajectory Comparison:**  
<img width="2235" height="595" alt="evolution" src="https://github.com/user-attachments/assets/614d3264-f6b5-4fbc-a7c8-ed65cb10f3fc" />

---
