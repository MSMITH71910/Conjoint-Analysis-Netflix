# Netflix Conjoint Analysis - Deployment Guide

This guide provides comprehensive instructions for deploying and running the Netflix Conjoint Analysis project.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 500MB free space

### Required Python Packages
All dependencies are listed in `requirements.txt`:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- statsmodels >= 0.12.0
- scikit-learn >= 1.0.0

## 🚀 Quick Deployment

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/MSMITH71910/Conjoint-Analysis-Netflix.git
cd Conjoint-Analysis-Netflix

# Run automated setup and analysis
python setup.py
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Run the complete analysis
4. Generate all outputs

### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/MSMITH71910/Conjoint-Analysis-Netflix.git
cd Conjoint-Analysis-Netflix

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run analysis
python netflix_real_data_analysis_fixed.py
```

## 📊 Analysis Options

### 1. Real Data Analysis (Primary)
```bash
python netflix_real_data_analysis_fixed.py
```
**Outputs:**
- `netflix_real_data_analysis.png` - Comprehensive visualizations
- `real_data_market_scenarios.csv` - Market scenario analysis
- Console output with detailed insights

### 2. Simulated Data Analysis (Comparison)
```bash
python netflix_conjoint_analysis.py
```
**Outputs:**
- `conjoint_analysis_results.png` - Simulated data visualizations
- `relative_importance.png` - Attribute importance chart
- `market_scenarios.csv` - Simulated scenario analysis

### 3. Advanced Demo Analysis
```bash
python demo.py
```
**Outputs:**
- `custom_scenarios.csv` - Extended scenario testing
- `revenue_analysis.csv` - Revenue optimization analysis
- Detailed console output with business insights

### 4. Interactive Jupyter Analysis
```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter notebook
jupyter notebook Netflix_Conjoint_Analysis.ipynb
```

## 🔧 Configuration Options

### Data Configuration
To use your own data, replace `netflix_customer_survey.csv` with your dataset. Ensure it has these columns:
- `customerid`: Unique customer identifier
- `NumberAccounts`: Number of simultaneous streams (1-6)
- `price`: Price point (8, 10, 12, 15, 18, 20)
- `ExtraContent`: Content type (Disney, HBO, Marvel, etc.)
- `ads`: Ad frequency (none, one_per_day, one_per_show)
- `selected`: Binary choice (0 or 1)

### Analysis Parameters
Modify these parameters in the analysis files:

```python
# In netflix_real_data_analysis_fixed.py
class NetflixRealDataAnalysis:
    def __init__(self, data_file='netflix_customer_survey.csv'):
        # Change data_file to use different dataset
        pass
    
    def simulate_market_scenarios(self):
        # Modify scenarios list to test different combinations
        scenarios = [
            {'name': 'Custom Scenario', 'NumberAccounts': 4, 'price': 16, 
             'ExtraContent': 'Disney', 'ads': 'none'},
            # Add more scenarios here
        ]
```

## 📁 Output Files

### Generated Files
After running the analysis, you'll find these files:

**Real Data Analysis:**
- `netflix_real_data_analysis.png` - 12-panel visualization dashboard
- `real_data_market_scenarios.csv` - Market scenario results

**Simulated Data Analysis:**
- `conjoint_analysis_results.png` - Part-worth utilities visualization
- `relative_importance.png` - Attribute importance pie chart
- `profiles.csv` - Experimental design profiles
- `responses.csv` - Simulated consumer responses
- `market_scenarios.csv` - Market scenario analysis

**Demo Analysis:**
- `custom_scenarios.csv` - Extended scenario testing
- `revenue_analysis.csv` - Revenue optimization results

### File Descriptions

#### netflix_real_data_analysis.png
Comprehensive 12-panel dashboard showing:
1. Selection rate by content type
2. Selection rate by price
3. Selection rate by number of accounts
4. Selection rate by ad frequency
5. Content part-worth utilities
6. Ads part-worth utilities
7. Relative attribute importance
8. Price sensitivity curve
9. Overall selection distribution
10. Accounts vs Price heatmap
11. Content vs Ads heatmap
12. Model performance metrics

#### real_data_market_scenarios.csv
Market scenario analysis with columns:
- `Scenario`: Scenario name
- `Accounts`: Number of accounts
- `Price`: Price point
- `Content`: Content type
- `Ads`: Ad frequency
- `Utility`: Calculated utility score
- `Choice_Probability`: Probability of selection
- `Appeal_Score`: 1-10 appeal rating

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'pandas'
```
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. Data File Not Found
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'netflix_customer_survey.csv'
```
**Solution:**
- Ensure you're in the correct directory
- Check that `netflix_customer_survey.csv` exists
- Use absolute path if necessary

#### 3. Memory Issues
```bash
MemoryError: Unable to allocate array
```
**Solution:**
- Ensure you have at least 4GB RAM available
- Close other applications
- Consider using a subset of data for testing

#### 4. Visualization Issues
```bash
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```
**Solution:**
```bash
# Install GUI backend
pip install PyQt5
# or
export DISPLAY=:0  # Linux with X11
```

### Environment-Specific Issues

#### Linux/Ubuntu
```bash
# If you get permission errors
sudo apt-get update
sudo apt-get install python3-venv python3-pip

# If matplotlib has issues
sudo apt-get install python3-tk
```

#### macOS
```bash
# If you get SSL certificate errors
/Applications/Python\ 3.x/Install\ Certificates.command

# If matplotlib has issues
brew install python-tk
```

#### Windows
```bash
# Use Command Prompt or PowerShell as Administrator
# If you get execution policy errors in PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 🔒 Security Considerations

### Data Privacy
- The included dataset is anonymized
- Customer IDs are randomized
- No personally identifiable information is included

### Code Security
- All dependencies are from trusted sources
- No external API calls or network requests
- All data processing is local

## 📈 Performance Optimization

### For Large Datasets
If working with datasets larger than 10,000 responses:

```python
# Modify the analysis to use sampling
import pandas as pd

# Sample data for faster processing
df_sample = df.sample(n=5000, random_state=42)

# Or use chunked processing
chunk_size = 1000
for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    # Process chunk
    pass
```

### Memory Optimization
```python
# Reduce memory usage
df = df.astype({
    'NumberAccounts': 'int8',
    'price': 'int8',
    'selected': 'int8'
})
```

## 🧪 Testing

### Validate Installation
```bash
# Test basic functionality
python -c "import pandas, numpy, matplotlib, seaborn, statsmodels, sklearn; print('All packages imported successfully')"

# Test analysis with sample data
python -c "
from netflix_real_data_analysis_fixed import NetflixRealDataAnalysis
netflix = NetflixRealDataAnalysis()
print('Analysis class initialized successfully')
"
```

### Run Test Analysis
```bash
# Quick test run (modify the analysis to use a subset)
python -c "
import pandas as pd
df = pd.read_csv('netflix_customer_survey.csv')
print(f'Data loaded: {df.shape}')
print(f'Selection rate: {df[\"selected\"].mean():.3f}')
"
```

## 🚀 Production Deployment

### For Production Use

1. **Data Validation**
   ```python
   # Add data validation
   def validate_data(df):
       required_columns = ['customerid', 'NumberAccounts', 'price', 'ExtraContent', 'ads', 'selected']
       assert all(col in df.columns for col in required_columns)
       assert df['selected'].isin([0, 1]).all()
       assert df['NumberAccounts'].between(1, 6).all()
       # Add more validations
   ```

2. **Error Handling**
   ```python
   # Wrap analysis in try-catch
   try:
       netflix = NetflixRealDataAnalysis()
       results = netflix.run_complete_analysis()
   except Exception as e:
       logging.error(f"Analysis failed: {str(e)}")
       # Handle error appropriately
   ```

3. **Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "netflix_real_data_analysis_fixed.py"]
```

## 📞 Support

### Getting Help
1. **Check this deployment guide** for common issues
2. **Review the README.md** for usage instructions
3. **Examine the code comments** for implementation details
4. **Open a GitHub issue** for bugs or feature requests

### Contact Information
- **GitHub**: MSMITH71910
- **Project**: https://github.com/MSMITH71910/Conjoint-Analysis-Netflix

---

**Last Updated**: July 2025  
**Version**: 1.0  
**Status**: Production Ready