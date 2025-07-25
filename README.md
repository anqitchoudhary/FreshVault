# FreshVault: Dynamic Price Optimization for Perishables

**Live Demo:** [**Click here to view the deployed application**](https://freshvault.streamlit.app/) 


---

## 📋 Project Overview

FreshVault is an intelligent web application designed to help grocery store managers minimize waste and maximize revenue by dynamically optimizing prices for perishable goods. The application uses a machine learning model to predict the optimal discount percentage for items nearing their expiry date, based on various factors like stock quantity, sales velocity, and seasonality.

The platform provides a dual-interface system:
* A **Store Manager Dashboard** for uploading inventory data, reviewing AI-powered discount suggestions, and managing active promotions.
* A **Customer-Facing Portal** where shoppers can view and explore all the currently available deals and discounts.

This project demonstrates an end-to-end data science workflow, from model training and evaluation to deployment as a user-friendly, interactive web application.

---

## ✨ Key Features

### Store Manager Dashboard:
* **Secure Login:** Role-based access for store managers.
* **AI-Powered Predictions:** Upload a CSV of inventory data to receive instant discount suggestions from the XGBoost model.
* **Review & Approve:** An interactive interface to approve or reject each discount suggestion individually.
* **Active Discount Management:** View all current promotions and remove them from the customer view at any time.
* **Data Download:** Download lists of approved items for record-keeping.
* **Sample Data:** Includes a downloadable sample CSV for easy testing and demonstration.

### Customer Portal:
* **Deal Discovery:** View all active, non-expired discounts in a clean, card-based layout.
* **Data-Rich Metrics:** See a summary of total deals, average discount, and the day's best offer.
* **Category Filtering:** Easily filter available discounts by product type (e.g., Fruit, Dairy, Meat).
* **Download Deals:** Customers can download a CSV of the current deals for their convenience.

---

## 🛠️ Technology Stack

* **Backend & ML:** Python
* **Web Framework:** Streamlit
* **Machine Learning Model:** XGBoost Regressor
* **Data Manipulation:** Pandas, NumPy
* **Model Persistence:** Scikit-learn, Joblib

---

## 📂 Project Structure

.├── 📄 approved_items.csv      # Stores the manager-approved discounts (acts as a simple DB)├── 📄 dashboard.py            # The main Streamlit application script├── 📄 encoders.pkl            # Saved label encoders for categorical data├── 📄 ml_model.pkl            # The final trained XGBoost model├── 📄 model.py                # Script for training the ML model from the dataset├── 📄 perishable_goods_pricing_dataset_large.csv  # The full 1000-item dataset used for training├── 📄 requirements.txt        # Python dependencies for the project├── 📄 sample_inventory.csv    # A small sample file for easy testing by users/recruiters└── 📄 README.md               # This file
---

## 🚀 Setup and Installation

To run this project on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/freshvault-streamlit-app.git](https://github.com/your-username/freshvault-streamlit-app.git)
cd freshvault-streamlit-app
2. Create and Activate a Virtual EnvironmentIt's recommended to use a virtual environment to keep dependencies isolated.# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesInstall all the necessary packages listed in requirements.txt.pip install -r requirements.txt
4. Train the Model (Optional)The repository already includes a pre-trained model (ml_model.pkl). However, if you want to retrain the model on the provided dataset, simply run:python model.py
This will generate new ml_model.pkl and encoders.pkl files.5. Run the Streamlit ApplicationYou are now ready to launch the FreshVault app!streamlit run dashboard.py
The application will open in your default web browser.📖 How to UseLogin: Start by logging in either as a Store Manager or a Customer. Use any username and password.As a Store Manager:Download the sample_inventory.csv file from the link provided in the app.Upload this file to the file uploader.Review the discount predictions generated by the model.Approve or reject items.Click "Confirm Final Approval" to make the deals live for customers.Navigate to the "Manage Active Discounts" section to remove items if needed.As a Customer:Instantly view all the deals that the manager has approved.Filter the deals by category.Enjoy the savings!</markdown>
