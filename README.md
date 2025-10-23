## 📌 Section 1: Project Overview

The Black-Box Optimization (BBO) Capstone Project is a machine learning challenge focused on optimizing unknown functions using limited queries. Each function is treated as a black box — its internal structure is hidden, and the only feedback available is the output returned for each submitted input.

The overall goal is to develop strategies that intelligently select input queries to maximize the output of these functions. This involves balancing exploration (trying new regions) and exploitation (focusing on known good areas), while working under uncertainty and with minimal data.

This challenge mirrors real-world machine learning tasks such as hyperparameter tuning, experimental design, and reinforcement learning, where decisions must be made with incomplete information and noisy feedback. It helps build skills in adaptive modeling, optimization, and technical communication — all essential for a career in data science or applied machine learning.

### Functions in Scope

| **Function**   | **Input Dimensionality** | **Output Type** | **Optimisation Goal** | **Sample Application Description** |
|----------------|--------------------------|------------------|------------------------|-------------------------------------|
| **Function 1** | 2D array                 | 1D array         | Maximise               | Detect contamination sources in a 2D radiation field using Bayesian optimisation to tune detection parameters. |
| **Function 2** | 2D array                 | 1D array         | Maximise               | Maximise noisy log-likelihood scores from a mystery ML model using Bayesian optimisation to avoid local optima. |
| **Function 3** | 3D array                 | 1D array         | Maximise               | Drug discovery: test compound combinations to minimise side effects (transformed to maximisation). |
| **Function 4** | 4D array                 | 1D array         | Maximise               | Optimise product placement across warehouses using ML approximations of costly biweekly calculations. |
| **Function 5** | 4D array                 | 1D array         | Maximise               | Maximise chemical process yield by tuning four input variables in a typically unimodal function. |
| **Function 6** | 5D array                 | 1D array         | Maximise               | Optimise a cake recipe with five ingredients to minimise negative scoring factors (transformed to maximisation). |
| **Function 7** | 6D array                 | 1D array         | Maximise               | Tune six ML hyperparameters to maximise model performance (e.g., accuracy or F1 score). |
| **Function 8** | 8D array                 | 1D array         | Maximise               | Tune eight ML hyperparameters to maximise validation accuracy in a high-dimensional, complex black-box function. |

---

## 📥 Section 2: Inputs and Outputs

Each week, the model receives past input–output data for each function and must submit one new query per function. The goal is to select queries that improve performance based on limited feedback.

### Inputs:
- Format: A vector of normalized values between 0 and 1  
- Dimensionality: Varies by function (2D to 8D)  
- Constraints:  
  - Each input must be within the range [0, 1]  
  - Only one query per function per week  
- Example Queries:  
  - Function 1: `0.793745-0.763745`  
  - Function 5: `0.352536-0.820938-0.794749-0.871774`  
  - Function 8: `0.036105-0.349529-0.029123-0.509092-0.904733-0.455832-0.319773-0.505309`

### Outputs:
- Format: A single scalar value returned by the black-box function  
- Type: 1D array (single float)  
- Meaning: Represents the performance or reward associated with the submitted input  
- Example Outputs:  
  - Function 5: `1074.7568746609543`  
  - Function 6: `-0.71`  
  - Function 8: `9.836`

## 🎯 Section 3: Challenge Objectives

The primary objective of the BBO Capstone Project is to **maximize** the output of eight unknown black-box functions. Each function represents a different real-world scenario, ranging from hyperparameter tuning to chemical yield optimization. The challenge is to identify high-performing input regions using limited feedback and no access to the internal structure of the functions.

### Optimization Goal:
- All eight functions are maximization tasks.
- Outputs are scalar performance signals based on submitted input vectors.

### Constraints and Limitations:
- **Limited Queries**: Only one query per function per week is allowed.
- **Unknown Function Structure**: The internal logic, gradients, and analytical form of each function are hidden.
- **Noisy Outputs**: Some functions return noisy or unstable outputs, making modeling and prediction more difficult.
- **Dimensionality**: Input spaces range from 2D to 8D, increasing complexity and risk of overfitting.
- **Delayed Feedback**: Outputs are returned after submission, requiring careful planning and iterative strategy updates.

The challenge is to build models that learn from sparse input–output data, adapt over time, and balance exploration of new regions with exploitation of known high-performing areas.

## 🧪 Section 4: Technical Approach

This section documents my evolving strategy across the first three query submissions of the BBO capstone project. As the challenge progresses, I continue refining my approach to better model the unknown functions and select high-performing queries.

### Week 1–2: GP-Based Exploration

In the initial rounds, I used **Gaussian Process (GP) regression** to model each function. I experimented with different kernels including Matern, RBF, and RationalQuadratic, and applied acquisition functions such as **Upper Confidence Bound (UCB)** and **Expected Improvement (EI)** to guide query selection.

- Candidate points were generated randomly across the input space.
- Kernel parameters were tuned heuristically based on model fit and acquisition behavior.
- Strategy focused on broad exploration to understand each function’s landscape.

### Week 3: SVM-Guided Filtering

In Week 3, I introduced **Support Vector Machines (SVMs)** to classify regions of the input space as “promising” or “not promising.” This helped filter out low-potential candidates before applying GP acquisition scores.

- Trained soft-margin SVMs using past input–output data.
- Used kernel SVMs to capture non-linear decision boundaries.
- Combined SVM confidence scores with GP predictions to select final queries.

This hybrid approach improved efficiency by reducing wasted queries and focusing on regions with higher likelihood of success.

### Exploration vs Exploitation Strategy

I balance exploration and exploitation dynamically based on function behavior:

- **Exploit**: For functions with strong prior outputs (e.g., Function 5 and Function 8), I sample near known peaks.
- **Explore**: For noisy or uncertain functions (e.g., Function 1 and Function 6), I use UCB and SVM filtering to explore new regions.
- **Mixed**: For functions with moderate or deceptive behavior (e.g., Function 3 and Function 7), I alternate strategies based on model confidence.

### Function-Specific Strategy Highlights

| Function    | Strategy Used                          | Notes |
|-------------|----------------------------------------|-------|
| Function 1  | GP + SVM filtering                     | Flat surface, cautious exploration |
| Function 2  | GP with UCB                            | Gradual slope, moderate exploitation |
| Function 3  | GP + kernel SVM                        | Noisy and nonlinear, adaptive modeling |
| Function 4  | GP + SVM filtering                     | Low output, filtered candidate space |
| Function 5  | GP + SVM exploitation                  | High output, targeted sampling |
| Function 6  | GP + SVM filtering                     | Sparse signal, avoid noisy traps |
| Function 7  | GP + alternating acquisition functions | Mixed behavior, adaptive strategy |
| Function 8  | GP + kernel SVM                        | High-dimensional, complex interactions |

### What Makes My Approach Unique

- I combine **Bayesian modeling** with **classification-based filtering** to guide query selection.
- I adapt strategies per function, rather than using a one-size-fits-all approach.
- I visualize input–output patterns and monitor model confidence to avoid overfitting.
- I treat this as a real-world optimization task — learning iteratively, tuning models, and balancing risk.

This evolving strategy helps me make informed decisions under uncertainty and prepares me for complex ML tasks in real-world scenarios.

