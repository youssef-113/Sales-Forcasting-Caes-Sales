# Car Sales Forecasting and Optimization

## Project Overview
This project aims to build a robust forecasting model that helps businesses predict future car sales trends. The project is divided into four milestones, each focusing on a different aspect of the data science pipeline.

## Member
- Youssef Bassiony 
## Project Structure
```
├── Milestone1/                    # Data Collection and Cleaning
│   ├── Car_sales_CleanData.csv    # Cleaned dataset
│   ├── Report exploration.doc     # Data exploration report
│   └── CarSales Data collection ,exploration ,cleaning.ipynb
│
├── MileStone2/                    # Data Analysis and Visualization
│   ├── Data Analysis and Visualization.ipynb
│   ├── DashBoard.ipynb
│   ├── Cars-sales-visual.pbix
│   └── milestone2.pdf
│
├── Milestone3/                    # Model Development
│   ├── Forecasting Model Development and Optimization.ipynb
│   ├── Random_Forest_Regressor.pkl
│   ├── income-model.ipynb
│   └── Forecasting Model Performance Report.docx
│
└── Milestone4/                    # Deployment
    ├── dep/
    │   ├── main/              
    │   │   ├── Forcasting.py
    │   │   ├── models/
    │   │   ├── Data/
    │   │   └── Pages/
    │   |   └── requirements.txt
    └── MlFlow/
    │   ├── Data/
    │   ├──Models/
    │   ├──models Graphs/ 
```

## Milestones

### Milestone 1: Data Collection and Cleaning
- Collected car sales data with features including:
  - Car details (make, model, engine, transmission, etc.)
  - Customer information (gender, annual income)
  - Sales information (price, date, dealer details)
- Performed data cleaning and preprocessing
- Created a clean dataset for analysis

### Milestone 2: Data Analysis and Visualization
- Conducted exploratory data analysis
- Created interactive dashboards
- Visualized sales trends and patterns
- Analyzed relationships between features
- Generated insights for business decisions

### Milestone 3: Model Development and Optimization
- Developed multiple forecasting models:
  - Random Forest Regressor
  - XGBoost
  - Decision Tree
- Performed feature engineering
- Optimized model parameters
- Evaluated model performance using:
  - R² (R-squared)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
- Selected Random Forest as the best performing model

### Milestone 4: Deployment
- Developed a Streamlit frontend with:
  - User-friendly interface
  - Real-time predictions
  - Visual feedback
  - Error handling
-Mlflow 

## Features
- Car price prediction based on multiple features
- Interactive web interface
- Real-time predictions
- Comprehensive data analysis
- Model performance monitoring

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Power BI

## Setup and Installation

Alternatively, you can use the `setup.sh` script (on Linux/macOS or Git Bash on Windows) to create the virtual environment and install dependencies:
```bash
chmod +x setup.sh       # Make sure it's executable
./setup.sh
```
After running the script, you still need to activate the virtual environment as prompted by the script.

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies from the root directory:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   Navigate to the Streamlit app directory and run the app.
   ```bash
   cd Milestone4/dep/main
   streamlit run Forcasting.py
   ```

5. Run MLflow experiments:
   Navigate to the MLflow directory and run your MLflow Python scripts (e.g., `train.py`, `predict.py`).
   For example, to run the training script (e.g., `train.py`):
   ```bash
   cd Milestone4/MLFLOW
   # Option 1: If you have an MLproject file
   mlflow run .
   # Option 2: Directly run the Python script
   # python train.py
   ```
   For running prediction scripts (e.g., `predict_arima.py` or `predict.py`):
   ```bash
   # Example for predict_arima.py:
   python predict_arima.py --model_path "path/to/your/arima_model" --steps 10
   # Example for predict.py (if it takes similar arguments):
   # python predict.py --model_path "path/to/your/model.pkl" --input_data "path/to/your/input_data.csv"
   ```
   To view the MLflow UI, run:
   ```bash
   mlflow ui
   ```
   Then open your browser to `http://localhost:5000`.



## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
