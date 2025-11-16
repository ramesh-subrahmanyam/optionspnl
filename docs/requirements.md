# Requirements

## Python Version
- **Python 3.12.4** (or compatible 3.12.x)

## Conda Environment
- Environment name: `position_analysis`

## Required Packages

### Core Data Science Libraries
- **pandas** >= 2.2.3 - Data manipulation and analysis
- **numpy** >= 1.26.4 - Numerical computing

### Financial Data and Options Pricing
- **yfinance** >= 0.2.65 - Yahoo Finance API for stock data
- **py_vollib** >= 1.0.1 - Black-Scholes options pricing and Greeks calculation

### Jupyter Notebook Environment
- **jupyter** >= 1.0.0 - Jupyter metapackage
- **notebook** >= 7.0.8 - Jupyter Notebook interface
- **ipykernel** >= 6.28.0 - IPython kernel for Jupyter
- **jupyterlab** >= 4.0.11 - JupyterLab interface (optional, for enhanced UI)

### Environment and Configuration
- **python-dotenv** - Load environment variables from .env file

### Optional Dependencies
- **finnhub-python** - Earnings calendar data (for manage_earnings_database.py)

## Installation Instructions

### Using Conda (Recommended)

```bash
# Create the conda environment
conda create -n position_analysis python=3.12.4

# Activate the environment
conda activate position_analysis

# Install core packages
conda install pandas=2.2.3 numpy=1.26.4 jupyter notebook jupyterlab

# Install pip packages
pip install yfinance==0.2.65 py_vollib==1.0.1 python-dotenv

# For earnings database management (optional)
pip install finnhub-python
```

### Using pip only

```bash
# Create virtual environment
python3.12 -m venv position_analysis_env
source position_analysis_env/bin/activate  # On Windows: position_analysis_env\Scripts\activate

# Install all packages
pip install pandas==2.2.3 numpy==1.26.4 yfinance==0.2.65 py_vollib==1.0.1 python-dotenv jupyter notebook jupyterlab

# For earnings database management (optional)
pip install finnhub-python
```

## Environment Setup

### Configure API Keys

Create a `.env` file in the project root for sensitive credentials:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Finnhub API key
# FINNHUB_API_KEY=your_actual_key_here
```

Get your Finnhub API key from: https://finnhub.io/

**Note**: The `.env` file is gitignored and should never be committed to version control.

## Verification

After installation, verify the setup:

```bash
# Activate environment
conda activate position_analysis  # or source position_analysis_env/bin/activate

# Check Python version
python --version  # Should show Python 3.12.4

# Check package installation
pip list | grep -E "(pandas|numpy|yfinance|py_vollib)"
```

## Notes

- The project uses **Black-Scholes Greeks** from py_vollib for options delta calculations
- **yfinance** is used for fetching historical stock prices, volatility, and market data
- All Jupyter notebooks require the **ipykernel** package to run properly
- The project is designed to work with broker-exported CSV files for positions and trades data
