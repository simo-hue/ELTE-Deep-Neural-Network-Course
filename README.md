# 🧠 ELTE Deep Network Developments — Course Materials & Study Resources

![University](https://img.shields.io/badge/University-ELTE_Budapest-0065BD?style=for-the-badge&logo=academia&logoColor=white)
![Program](https://img.shields.io/badge/Program-EIT_Digital_MSc_(AUSIR)-blueviolet?style=for-the-badge)
![Semester](https://img.shields.io/badge/Semester-2026%2F2027_Fall-brightgreen?style=for-the-badge)
![Language](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge)

> **The most comprehensive open-source study resource for the Deep Network Developments course at [Eötvös Loránd University (ELTE)](https://www.elte.hu/en/), Budapest.**
> Labs, exercises, practice worksheets, LeetCode-style challenges, and hands-on neural network implementations — all in one place.

---

## 📖 Table of Contents

- [About the Course](#-about-the-course)
- [Topics Covered](#-topics-covered)
- [Course Schedule](#-course-schedule)
- [Grading & Assessment](#-grading--assessment)
- [Repository Structure](#-repository-structure)
- [Progress Tracker](#-progress-tracker)
- [Tech Stack & Prerequisites](#-tech-stack--prerequisites)
- [External Resources](#-external-resources)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 📚 About the Course

| Detail | Info |
|---|---|
| **Course Name** | Deep Network Developments |
| **University** | Eötvös Loránd University (ELTE), Budapest, Hungary |
| **Faculty** | Faculty of Informatics |
| **Program** | EIT Digital Master School (AUSIR — Autonomous Systems) |
| **Semester** | 2026/2027 — 1st Semester (Fall) |
| **Lecturer** | Viktor VARGA (vv@inf.elte.hu) |
| **Lectures** | Thursdays from 17:45, South Building 0.821 (in-person recommended) |
| **Practices** | 90 min/week, in-person (compulsory — max 4 absences allowed) |
| **Communication** | Teams group (join code: `ejqdiwz`) + Moodle |

This course introduces students to the **fundamental theory of neural networks** and the popular software libraries used to implement them. Starting from linear and logistic regression, it builds through multi-layer perceptrons, convolutional networks, recurrent networks, attention mechanisms, transformers, and unsupervised learning — providing both the mathematical foundations and practical coding skills.

### Practice Leaders

| Name | Email |
|---|---|
| Zsolt CSIBI | vrnf2j@inf.elte.hu |
| Áron Bálint JANIK | f5z688@inf.elte.hu |
| Imre MOLNÁR | imremolnar@inf.elte.hu |
| Vilmos Andor SZIRMAI | szvilmos@inf.elte.hu |
| Tamás TAKÁCS | tamastheactual@inf.elte.hu |

---

## 🧠 Topics Covered

- **Linear Regression** — Foundations of supervised learning, loss functions, gradient descent
- **Logistic Regression** — Binary classification, sigmoid function, decision boundaries
- **Under-/Over-fitting** — Bias-variance tradeoff, hyperparameter tuning
- **Artificial Neuron Model** — Perceptron, activation functions
- **Multi-Layer Perceptron (MLP)** — Feedforward networks, metrics, evaluation
- **Backpropagation Algorithm** — Gradient computation, chain rule, weight updates
- **Mitigating Over-fitting** — Regularization techniques (dropout, L1/L2, data augmentation)
- **Convolutional Neural Networks (CNNs)** — Convolution, pooling, feature extraction
- **Unstable Gradients** — Vanishing/exploding gradients, batch normalization
- **Transfer Learning** — Pre-trained models, fine-tuning, domain adaptation
- **Recurrent Neural Networks (RNNs)** — Sequential data, LSTM, GRU
- **Attention-Based Models** — Self-attention, multi-head attention mechanisms
- **Natural Language Processing (NLP)** — Text processing, embeddings, word2vec
- **Transformer Model** — Encoder-decoder architecture, positional encoding
- **Unsupervised Learning** — Autoencoders, generative models with neural networks

---

## 🗓️ Course Schedule

> **Semester**: 2026/2027 Fall (preliminary — subject to changes)

| Week | Date | Lecture / Assessment |
|------|------|---------------------|
| 1 | 10 September | Introduction |
| 2 | 17 September | Linear Regression |
| 3 | 24 September | Logistic Regression |
| 4 | 1 October | Under-/Over-fitting, Hyperparameters, Artificial Neuron Model |
| 5 | 8 October | Multi-Layer Perceptron Model, Metrics |
| 6 | 15 October | Backpropagation Algorithm |
| 7 | 22 October | Mitigating Over-fitting, Convolutional Neural Networks |
| 8 | 29 October | *Autumn break — No lecture* |
| 9 | 5 November | **Theory Test 1** + Lecture: Unstable Gradients, Transfer Learning |
| 10 | 12 November | Recurrent Neural Networks *(HW normal deadline ~15 Nov)* |
| 11 | 19 November | **Theory Test 1 Retake** + Lecture: Attention-Based Models *(HW final deadline ~22 Nov)* |
| 12 | 26 November | Basics of NLP, The Transformer Model |
| 13 | 3 December | **Coding Test** *(probable date, from 17:45 — no lecture)* |
| 14 | 10 December | Unsupervised Learning with Neural Networks |

### Exam Period

| Assessment | Probable Date |
|---|---|
| Theory Test 2 | 17 December (Thursday), from 18:00 |
| Coding Test Retake | 11 January (Monday), from 17:00 |
| Theory Test 2 Retake | 14 January (Thursday), from 18:00 |
| Oral Exams | Multiple occasions between 14 Dec – 29 Jan |

---

## 📊 Grading & Assessment

### Point Breakdown (Total: 80 points)

| Component | Points | Minimum Required |
|---|---|---|
| **Theory Test 1** | 15 pts | 7 pts (required for Coding Test) |
| **Theory Test 2** | 10 pts | 5 pts (required for Oral Exam) |
| **Homework Assignment (HW)** | 5 pts (if submitted by normal deadline) | Must be accepted (required for Coding Test) |
| **Coding Test** | 35 pts | 17 pts (required for passing grade) |
| **Oral Exam** | 15 pts | 7 pts (required for passing grade) |

### Grade Thresholds

| Points | Grade |
|---|---|
| 64 – 80 | **5** (Excellent) |
| 54 – 63 | **4** (Good) |
| 44 – 53 | **3** (Satisfactory) |
| 36 – 43 | **2** (Pass) |
| 0 – 35 | **1** (Fail) |

> ⚠️ If minimum requirements for any individual component are not met, the final grade is automatically **1 (Fail)**.

### Homework Assignment Rules

- Submit via **Moodle** assignment submission page.
- **Normal deadline**: ~15 November → earns 5 points if accepted.
- **Final deadline**: ~22 November → late submission loses the 5 points but still allows Coding Test participation.
- HW is **accept/reject** — no partial grading.
- Submissions are checked with **ML-based plagiarism detection** (not fooled by variable renaming, whitespace changes, etc.).
- Enable Moodle email notifications for feedback on submissions.

### Coding Test Rules

- Conducted in a **computer lab, closed circumstances**, no internet access.
- **180 minutes** duration.
- Must present official **photo ID** (passport or ID card).
- Allowed materials: own HW solution, all non-video course materials, software library documentation.
- One retake available during exam period.

### Oral Exam Details

- ~15 minutes, during exam period.
- Must present official **photo ID**.
- Maximum **2 attempts** per semester.

### AI Usage Policy

- ❌ **Prohibited** in closed examinations (Coding Test, Theory Tests, Oral Exams).
- ⚠️ **Not prohibited but discouraged** for Homework — since the Coding Test has no internet, relying on AI for HW is counterproductive.
- Unauthorized circumvention → **immediate exclusion** from the course.

---

## 📂 Repository Structure

```
ELTE-Deep-Neural-Network-Course/
├── README.md
├── PREREQUISITES/
│   ├── first_week_practise.ipynb        # Pre-course Python warm-up exercises
│   ├── self_evaluation.ipynb            # Self-assessment notebook
│   └── img.png                          # Supporting image
├── LAB/
│   ├── 1/                               # Week 1 — The Reading Room
│   │   ├── w01-the-reading-room.ipynb   # Practice worksheet (my solutions)
│   │   ├── teacher_solution.ipynb       # Official teacher solution
│   │   ├── Python, and What It Costs.pdf  # Reading material
│   │   ├── inputs.json                  # Exercise data
│   │   ├── part1_submission.csv         # Submission part 1
│   │   ├── part2_submission.csv         # Submission part 2
│   │   ├── part3_submission.csv         # Submission part 3
│   │   └── LEET CODE/                   # Bonus coding challenges
│   │       ├── 1.py – 6.py             # LeetCode-style solutions
│   ├── 2/                               # Week 2 — Objects That Remember
│   │   ├── dnd26autumn_practice_week2_worksheet.ipynb  # Practice worksheet
│   │   ├── solution.ipynb               # My solution
│   │   ├── Objects That Remember.pdf    # Reading material
│   │   ├── AT_HOME/                     # At-home exercises
│   │   ├── AT_UNI/                      # In-class exercises
│   │   └── LEET CODE/                   # Bonus coding challenges
│   ├── 3/                               # Week 3 — Series and Masks (NumPy)
│   │   ├── w03-series-and-masks.ipynb   # Practice worksheet (my solutions)
│   │   ├── dnd26autumn_practice_week3_worksheet.ipynb  # Original worksheet
│   │   ├── NumPy Arrays in One Dimension.pdf  # Reading material
│   │   ├── level.npy / wind.npy         # NumPy data files
│   │   ├── queries.json                 # Query data
│   │   ├── part1_submission.csv         # Submission part 1
│   │   ├── part2_submission.csv         # Submission part 2
│   │   ├── part3_submission.csv         # Submission part 3
│   │   ├── PRE-WORK/                    # Pre-lab preparation
│   │   └── LEET-CODE/                   # Bonus coding challenges
│   └── 4/                               # Week 4 — (upcoming)
└── tmp.py                               # Temporary scratch file
```

---

## ✅ Progress Tracker

### Prerequisites
- [x] First week practice (Python warm-up)
- [x] Self-evaluation notebook

### Labs
- [x] Lab 1 — The Reading Room (Python fundamentals + LeetCode challenges)
- [x] Lab 2 — Objects That Remember (OOP + classes)
- [x] Lab 3 — Series and Masks (NumPy arrays, masking, queries)
- [ ] Lab 4
- [ ] Lab 5
- [ ] Lab 6
- [ ] Lab 7
- [ ] Lab 8
- [ ] Lab 9
- [ ] Lab 10
- [ ] Lab 11
- [ ] Lab 12

### Assessments
- [ ] Theory Test 1 (Week 9 — 5 November)
- [ ] Homework Assignment (normal deadline ~15 November)
- [ ] Theory Test 2 (Exam period — ~17 December)
- [ ] Coding Test (Week 13 — ~3 December)
- [ ] Oral Exam (Exam period)

---

## 🛠️ Tech Stack & Prerequisites

| Tool / Library | Purpose |
|---|---|
| **Python 3.12+** | Primary programming language |
| **NumPy** | Numerical computing, array operations, MLP implementation from scratch |
| **PyTorch** | Deep learning framework for neural network implementation |
| **Jupyter Notebook** | Interactive lab exercises and worksheets |
| **Matplotlib** | Data visualization and plotting |

### Getting Started

```bash
# Clone the repository
git clone https://github.com/simo-hue/ELTE-Deep-Neural-Network-Course.git

# Navigate to the project
cd ELTE-Deep-Neural-Network-Course

# Install dependencies
pip install numpy torch matplotlib jupyter
```

---

## 🔗 External Resources

| Resource | Link |
|---|---|
| **ELTE Faculty of Informatics** | [inf.elte.hu](https://www.inf.elte.hu/en/) |
| **Course Materials (Moodle)** | Available via ELTE Moodle (login required) |
| **Teams Group** | Join with code: `ejqdiwz` |
| **PyTorch Documentation** | [pytorch.org/docs](https://pytorch.org/docs/stable/) |
| **NumPy Documentation** | [numpy.org/doc](https://numpy.org/doc/stable/) |

---

## ⚠️ Disclaimer

> This repository is intended **for educational purposes only**.
> All lecture materials, slides, worksheets, PDFs, and course content belong to their respective authors and to **Eötvös Loránd University (ELTE)**, Budapest.
> Personal notes, exercise solutions, and code implementations are my own work.
> This repository is not officially affiliated with or endorsed by ELTE or the course staff.
>
> **Important**: Do not use these solutions to circumvent academic integrity policies. The coding test is conducted without internet access, so genuine understanding is essential.

---

## 🌟 Star This Repo

If you find these materials helpful for your studies, please consider giving this repository a ⭐ — it helps other ELTE students discover it!

---

<p align="center">
  Made with 🧠 by <a href="https://simo-hue.github.io">Simone Mattioli</a> for Deep Network Developments students at ELTE Budapest
</p>