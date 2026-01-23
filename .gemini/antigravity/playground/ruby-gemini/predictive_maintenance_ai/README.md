# ✈️ Aero-Guard: Predictive Maintenance AI

![Model Loss Curve](loss_curve.png)

A predictive maintenance system for jet engines using NASA's CMAPSS dataset. This project uses Deep Learning (LSTM) to predict the Remaining Useful Life (RUL) of an engine based on sensor data.

## 🚀 Features

*   **Deep Learning Model:** An LSTM (Long Short-Term Memory) network trained on time-series sensor data.
*   **Real-time Dashboard:** A Streamlit app that simulates live engine data and visualizes RUL predictions.
*   **PyTorch Implementation:** Built with PyTorch for robust and flexible model training.

## 🛠️ Tech Stack

*   **Python 3.13**
*   **PyTorch** (Model Training)
*   **Streamlit** (Dashboard)
*   **Pandas & NumPy** (Data Processing)
*   **Plotly** (Visualization)

## 📂 Project Structure

```
predictive_maintenance_ai/
├── data/               # NASA CMAPSS Dataset
├── train_model.py      # Data processing & LSTM training script
├── dashboard.py        # Streamlit dashboard application
├── nasa_engine_model.pth # Trained PyTorch model
├── loss_curve.png      # Training loss visualization
└── README.md           # This file
```

## 🏃‍♂️ How to Run

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib scikit-learn torch torchvision streamlit plotly
    ```

2.  **Train the Model:**
    ```bash
    python train_model.py
    ```
    This will generate `nasa_engine_model.pth` and `loss_curve.png`.

3.  **Launch the Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```

## 📊 Model Performance

The model is trained for 100 epochs. The loss curve above shows the convergence of the Mean Squared Error (MSE) loss over the training period.

## 🧠 Architecture

```mermaid
graph LR
    A[Sensors] --> B[Data Ingestion]
    B --> C[Sliding Window (50)]
    C --> D[LSTM Layer 1]
    D --> E[LSTM Layer 2]
    E --> F[RUL Prediction]
```
