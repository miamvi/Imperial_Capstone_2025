## 📌 Section 1: Project Overview

The Black-Box Optimization (BBO) Capstone Project is a machine learning challenge focused on optimizing unknown functions using limited queries. Each function is treated as a black box — its internal structure is hidden, and the only feedback available is the output returned for each submitted input.

The overall goal is to develop strategies that intelligently select input queries to maximize the output of these functions. This involves balancing exploration (trying new regions) and exploitation (focusing on known good areas), while working under uncertainty and with minimal data.

The BBO capstone has been a great way to sharpen my machine learning skills in a realistic setting. It’s helped me learn how to make smart decisions with limited data — something that comes up all the time in real-world projects like tuning models or running experiments. I’ve also had the chance to work with techniques like Gaussian Processes and SVMs, and combine them into a strategy that adapts over time.

Beyond the technical side, it’s taught me how to explain my thinking clearly — documenting my approach, justifying choices, and reflecting on results. That’s a big part of working in data science, whether I’m collaborating with a team or sharing work with future employers.

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

---

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

---

## 🧪 Section 4: Technical Approach

This section documents my evolving strategy across the first three query submissions of the BBO capstone project. As the challenge progresses, I continue refining my approach to better model the unknown functions and select high-performing queries.

### Week 1–2: GP-Based Exploration
- Used **Gaussian Process (GP) regression** with kernels (Matern, RBF, RationalQuadratic).  
- Applied acquisition functions (**UCB**, **EI**) to balance exploration and exploitation.  
- Candidate points generated randomly across the input space.  

### Week 3: SVM-Guided Filtering
- Introduced **Support Vector Machines (SVMs)** to classify regions as promising vs not promising.  
- Combined SVM confidence scores with GP predictions to select final queries.  
- Improved efficiency by reducing wasted queries.  

### Exploration vs Exploitation Strategy
- **Exploit**: Sample near known peaks (Functions 5, 8).  
- **Explore**: Use UCB + SVM filtering for noisy functions (Functions 1, 6).  
- **Mixed**: Alternate strategies for moderate functions (Functions 3, 7).  

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

---

## 📖 Section 5: Technical Justification & References

My current black‑box optimisation (BBO) approach is built around Gaussian Process (GP) surrogates combined with acquisition functions like Expected Improvement (EI) and Upper Confidence Bound (UCB). This choice is justified because GPs provide a principled way to model uncertainty, while acquisition functions balance exploration and exploitation. I also experimented with neural surrogates (MLPRegressor) to capture non‑stationary behaviour. These decisions are supported by free tutorials such as Kelta’s Datacamp article on mastering Bayesian optimisation, which explains why acquisition functions are effective, and by the scikit‑learn documentation, which provides reliable implementations of GP regression and kernels.

To make my reasoning clear, I document the foundations of my approach here:

### Main Justification
- **Gaussian Processes + Acquisition Functions**: Supported by free tutorials like [Datacamp’s “Mastering Bayesian Optimisation”](https://www.datacamp.com/blog/mastering-bayesian-optimization-in-data-science).  
- **SVM Filtering**: Inspired by scikit-learn’s open documentation on [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) and [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).  
- **Neural Surrogates**: Reinforced by Michael Nielsen’s free book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).  
- **Incremental Refinement Mindset**: Inspired by Andrej Karpathy’s blog [“What I learned from competing against a ConvNet on ImageNet”](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/).  
- **Frameworks**: Guided by free tutorials like [PyTorch’s 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [TensorFlow Playground](https://playground.tensorflow.org).

### How I Document Choices
- **README.md**: Summarises design choices and links to free resources.  
- **References.md**: Contains a full list of free/open references (blogs, docs, tutorials, YouTube).  
- **Code Comments**: Explain why specific kernels, surrogates, or acquisitions were chosen.  
- **Notebooks**: Include plots and tables showing how theory translates into practice
