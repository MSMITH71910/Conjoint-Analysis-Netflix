# Netflix Conjoint Analysis

A comprehensive conjoint analysis implementation using **real Netflix customer survey data** to address key business challenges:

1. **Increase the number of subscriptions**
2. **Increase the price** (without losing too many subscribers)
3. **Add revenue streams**
4. **Add new markets**

## 📊 Study Design & Real Data

### Dataset Overview
- **Real customer survey data**: 3,000 responses from 311 customers
- **Choice-based conjoint**: Customers selected preferred Netflix packages
- **Balanced experimental design**: Each customer evaluated multiple scenarios

### Attributes and Levels (From Real Data)
- **Content**: Disney, HBO, Marvel, Prime Originals, Soccer, Less Content
- **Number of Accounts**: 1, 2, 3, 4, 5, 6 simultaneous streams
- **Price**: $8, $10, $12, $15, $18, $20
- **Ads**: None, One per day, One per show

### Key Findings from Real Data
- **Selection Rate**: 49.8% (balanced choice data)
- **Model Performance**: R² = 0.075, highly significant (p < 0.001)
- **Strong Statistical Significance**: All key coefficients significant

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/MSMITH71910/Conjoint-Analysis-Netflix.git
cd Conjoint-Analysis-Netflix
python setup.py  # Automated setup and analysis
```

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Analysis Options

#### 1. Real Data Analysis (Recommended)
```python
python netflix_real_data_analysis_fixed.py
```

#### 2. Simulated Data Analysis (For Comparison)
```python
python netflix_conjoint_analysis.py
```

#### 3. Interactive Analysis
```bash
jupyter notebook Netflix_Conjoint_Analysis.ipynb
```

## 📁 File Structure

```
Conjoint-Analysis-Netflix/
├── netflix_real_data_analysis_fixed.py    # Main real data analysis
├── netflix_conjoint_analysis.py           # Simulated data analysis
├── netflix_customer_survey.csv            # Real customer survey data
├── Netflix_Conjoint_Analysis.ipynb        # Interactive Jupyter notebook
├── demo.py                                # Advanced scenario testing
├── run_analysis.py                        # Quick analysis runner
├── setup.py                              # Automated setup script
├── requirements.txt                       # Python dependencies
├── README.md                             # This file
├── DEPLOYMENT.md                         # Deployment guide
├── SUMMARY.md                            # Executive summary
├── netflix_real_data_analysis.png       # Real data visualizations
├── real_data_market_scenarios.csv       # Real data scenario analysis
└── venv/                                 # Virtual environment
```

## 🔍 Analysis Results (Real Data)

### Part-Worth Utilities
**Content Preferences** (relative to Disney baseline):
- Disney: 0.000 (baseline)
- HBO: +0.005 (slight preference)
- Marvel: -0.022 (slight negative)
- Prime Originals: -0.024 (slight negative)
- Soccer: -0.103 (moderate negative)
- Less Content: -0.186 (strong negative)

**Account Preferences**:
- Coefficient: +0.045 per additional account
- Strong positive preference for multi-account plans
- Each additional account increases selection probability

**Price Sensitivity**:
- Coefficient: -0.016 per dollar
- Moderate price sensitivity
- Each $1 increase reduces utility by 0.016

**Ad Preferences** (relative to no ads):
- No Ads: 0.000 (baseline - most preferred)
- One per day: -0.024 (slight negative)
- One per show: -0.133 (strong negative)

### Relative Importance
1. **Number of Accounts**: 30.6% (most important)
2. **Price**: 25.7% (second most important)
3. **Content**: 25.7% (tied for second)
4. **Ads**: 18.0% (least important but still significant)

### Top Market Scenarios (Real Data)
1. **Premium All**: HBO + 6 accounts + $20 + no ads (65.0% choice probability)
2. **Disney Premium**: Disney + 4 accounts + $15 + no ads (64.7% choice probability)
3. **Marvel Family**: Marvel + 6 accounts + $18 + one ad/day (64.6% choice probability)

## 💡 Business Insights (From Real Data)

### 1. Content Strategy
- **Disney and HBO** perform best (Disney is baseline, HBO slightly better)
- **Sports content** has limited appeal (-0.103 utility)
- **Content differentiation** is crucial (25.7% of decision)
- **Avoid "less content"** positioning (-0.186 utility penalty)

### 2. Pricing Strategy
- **Moderate price sensitivity** (-0.016 per dollar)
- **Sweet spot**: $12-$18 range based on utility curves
- **Premium pricing viable** with right content/account mix
- **Price increases** must be bundled with value additions

### 3. Account Strategy
- **Most important factor** (30.6% of decision)
- **Strong preference** for multi-account plans (+0.045 per account)
- **Family plans** are key differentiator
- **6-account tiers** command premium pricing

### 4. Advertising Strategy
- **Ad-free strongly preferred** (baseline)
- **One ad per show** significantly hurts appeal (-0.133)
- **One ad per day** more acceptable (-0.024)
- **Ad-supported tiers** viable for price-sensitive segments

## 🎯 Addressing Business Problems

### Increase Subscriptions
- **Lead with multi-account value** (30.6% importance)
- **Emphasize premium content** (Disney/HBO positioning)
- **Competitive pricing** at $12-$15 entry point
- **Family-focused marketing**

### Increase Price
- **Gradual increases** to $15-$18 range
- **Bundle with additional accounts** (high utility)
- **Premium content justification** (HBO/Disney)
- **Avoid aggressive pricing** (moderate sensitivity)

### Add Revenue Streams
- **Premium content tiers**: HBO/Disney at $15-$20
- **Family plan upsells**: 4-6 account tiers
- **Ad-supported options**: Lower price with ads
- **Sports packages**: Niche but viable at premium

### New Markets
- **Replicate account-focused strategy**
- **Adapt content mix** to local preferences
- **Use pricing insights** for market entry
- **Family plan emphasis** across cultures

## 📊 Visualizations

The analysis generates comprehensive visualizations:

1. **Selection rates** by each attribute level
2. **Part-worth utilities** for all attributes
3. **Relative importance** breakdown
4. **Price sensitivity curves**
5. **Heatmaps** showing attribute interactions
6. **Model performance metrics**

## 🔧 Customization

### Testing New Scenarios
```python
from netflix_real_data_analysis_fixed import NetflixRealDataAnalysis

# Initialize analysis
netflix = NetflixRealDataAnalysis()
netflix.run_complete_analysis()

# Test custom scenario
utility = netflix.predict_scenario_utility(
    accounts=4, 
    price=16, 
    content='Disney', 
    ads='none'
)
print(f"Scenario utility: {utility:.3f}")
```

### Modifying Analysis
- Update `netflix_customer_survey.csv` with new data
- Modify attribute levels in the analysis code
- Add new scenarios in the simulation section

## 📚 Technical Details

### Statistical Models
- **OLS Regression**: For part-worth utility estimation
- **Logistic Regression**: For choice probability prediction
- **Dummy Coding**: With first category as baseline
- **Multicollinearity Handling**: Drop-first approach

### Model Performance
- **R-squared**: 0.075 (good for choice data)
- **F-statistic**: 27.06 (highly significant)
- **Sample Size**: 3,000 responses (robust)
- **Significance**: All key coefficients p < 0.05

### Design Properties
- **Real customer data**: Authentic preferences
- **Balanced design**: Equal representation of levels
- **Choice-based**: Realistic decision context
- **Large sample**: 311 customers, 3,000 responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Review the analysis code for implementation details
- Check DEPLOYMENT.md for setup instructions

## 🏆 Project Highlights

- ✅ **Real customer data** (3,000 responses)
- ✅ **Comprehensive analysis** (OLS + Logistic regression)
- ✅ **Business-focused insights** (actionable recommendations)
- ✅ **Professional visualizations** (12 charts + heatmaps)
- ✅ **Scenario simulation** (8 market scenarios tested)
- ✅ **Statistical rigor** (significance testing, model validation)
- ✅ **Production-ready code** (error handling, documentation)

---

**Author**: MSMITH71910  
**Date**: July 2025  
**Status**: Production Ready

This analysis provides Netflix with data-driven insights to optimize their subscription strategy, pricing model, and market expansion plans based on real customer preferences.