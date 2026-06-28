# 🌾 Rice Variety Classification System using PyTorch & Streamlit

This is an end-to-end Deep Learning application that classifies rice varieties into **Jasmine** or **Gonen** based on their geometric and structural features. The model is built using **PyTorch**, trained on Kaggle's Rice Type Classification dataset, and deployed via an interactive web interface using **Streamlit**.

---

## 🎯 Project Overview
In the agricultural industry, automated classification of grain types is crucial for quality control and processing. This project takes 10 distinct morphological attributes of a rice grain and predicts its specific type with high confidence.

### Key Features:
- **PyTorch Deep Learning Engine:** A Neural Network architecture trained with structured features using optimized weights.
- **Real-time Normalization:** Seamless data preprocessing on the fly before passing values to the model.
- **Interactive Streamlit UI:** A clean, two-column responsive dashboard for seamless user testing.
- **Dynamic Confidence Level:** Displays the prediction percentage utilizing Sigmoid activation.

---

## 📊 Dataset Features Included
The system accepts the following 10 parameters for a single sample:
1. **Area:** Total surface area of the grain.
2. **Major Axis Length:** The longest diameter of the grain.
3. **Minor Axis Length:** The shortest diameter of the grain.
4. **Eccentricity:** Measure of how elongated the grain is.
5. **Convex Area:** Area of the smallest convex polygon enclosing the grain.
6. **Equiv Diameter:** Diameter of a circle with the same area as the grain.
7. **Extent:** Ratio of pixels in the grain region to pixels in the total bounding box.
8. **Perimeter:** The length of the grain's outer boundary.
9. **Roundness:** Sharpness or circular perfection of the grain.
10. **Aspect Ratio:** Ratio of major axis length to minor axis length.

---

## 🛠️ Tech Stack & Dependencies
- **Language:** Python
- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`)
- **Web App Framework:** Streamlit
- **Data Manipulation:** NumPy, Pandas, Scikit-Learn

---

## 🚀 How to Run Locally

### 1. Clone the repository:
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/rice-variety-classifier-pytorch.git](https://github.com/YOUR_GITHUB_USERNAME/rice-variety-classifier-pytorch.git)
cd rice-variety-classifier-pytorch

### 2. Install dependencies:
```markdown
```bash
pip install -r requirements.txt

3. Run the Streamlit App:
```bash
streamlit run main.py

💡 Preprocessing & Inference LogicThe features are scaled dynamically using the maximum values derived from the training dataset. This ensures that the single test inputs map exactly to the scale ($0$ to $1$) on which the neural network was optimized, preventing performance mismatch during live execution.
