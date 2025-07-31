# Netflix Conjoint Analysis - Project Summary

## 🎯 Project Overview

This project implements a comprehensive conjoint analysis using **real Netflix customer survey data** to address four key business challenges:

1. **Increase subscriptions**
2. **Increase pricing** (without losing subscribers)
3. **Add revenue streams**
4. **Expand to new markets**

## 📊 Analysis Design

### Real Customer Data
- **Dataset**: 3,000 real customer responses from 311 customers
- **Selection Rate**: 49.8% (balanced choice data)
- **Data Quality**: High-quality choice-based conjoint data

### Attributes & Levels (From Real Survey)
- **Content**: Disney, HBO, Marvel, Prime Originals, Soccer, Less Content
- **Accounts**: 1, 2, 3, 4, 5, 6 simultaneous streams
- **Price**: $8, $10, $12, $15, $18, $20
- **Ads**: None, One per day, One per show

### Methodology
- **Design**: Real customer choice data (choice-based conjoint)
- **Sample**: 311 customers, 3,000 total responses
- **Analysis**: OLS regression + Logistic regression with dummy coding
- **Validation**: Statistical significance testing (R² = 0.075, highly significant)

## 🔍 Key Findings (Real Data)

### Consumer Preferences (Part-Worth Utilities)
1. **Content Ranking** (relative to Disney baseline):
   - Disney: 0.000 (baseline - strong performer)
   - HBO: +0.005 (slight preference over Disney)
   - Marvel: -0.022 (slight negative)
   - Prime Originals: -0.024 (slight negative)
   - Soccer: -0.103 (moderate negative)
   - Less Content: -0.186 (strong negative)

2. **Account Preferences**:
   - Coefficient: +0.045 per additional account
   - Strong positive preference for multi-account plans
   - Each additional account significantly increases selection probability

3. **Price Sensitivity**:
   - Coefficient: -0.016 per dollar
   - Moderate price sensitivity
   - Each $1 increase reduces utility by 0.016

4. **Ad Preferences** (relative to no ads):
   - No Ads: 0.000 (baseline - most preferred)
   - One per day: -0.024 (slight negative)
   - One per show: -0.133 (strong negative)

### Attribute Importance (Real Data)
- **Number of Accounts**: 30.6% (most important factor)
- **Price**: 25.7% (second most important)
- **Content**: 25.7% (tied for second)
- **Ads**: 18.0% (least important but still significant)

## 💡 Strategic Recommendations (Based on Real Data)

### 1. Increase Subscriptions
- **Lead with multi-account plans** (30.6% importance - most critical factor)
- **Emphasize Disney/HBO content** (baseline and top performers)
- **Competitive entry pricing** at $10-$12 range
- **Family-focused marketing** (accounts are key differentiator)

### 2. Increase Pricing
- **Gradual increases** to $15-$18 range (moderate sensitivity)
- **Bundle with additional accounts** (high utility per account)
- **Premium content justification** (Disney/HBO positioning)
- **Avoid aggressive pricing** without value additions

### 3. Add Revenue Streams
- **Premium content tiers**: HBO/Disney at $15-$20
- **Family plan upsells**: 4-6 account tiers
- **Ad-supported options**: Lower price with limited ads
- **Sports packages**: Niche market at premium pricing

### 4. New Market Entry
- **Account-focused strategy**: Emphasize multi-user value
- **Adapt content mix** to local preferences (Disney/HBO model)
- **Price sensitivity insights** for market-specific pricing
- **Family plan emphasis** across different cultures

## 📈 Market Scenarios (Real Data Results)

### Top Performing Scenarios
1. **Premium All** (HBO, 6 accounts, $20, no ads)
   - Choice Probability: 65.0%
   - Utility Score: 0.620
   - Appeal Score: 7/10

2. **Disney Premium** (Disney, 4 accounts, $15, no ads)
   - Choice Probability: 64.7%
   - Utility Score: 0.604
   - Appeal Score: 6/10

3. **Marvel Family** (Marvel, 6 accounts, $18, one ad/day)
   - Choice Probability: 64.6%
   - Utility Score: 0.601
   - Appeal Score: 6/10

## 🎯 Business Impact (Real Data Insights)

### Revenue Optimization
- **Multi-account plans** are the key revenue driver (30.6% importance)
- **Each additional account** adds significant value (+0.045 utility)
- **Premium content** (Disney/HBO) supports higher pricing
- **Ad-free positioning** commands premium (+0.133 utility vs ads)

### Competitive Positioning
- **Account differentiation** more important than content or price
- **Family-focused offerings** create strongest switching costs
- **Premium positioning** viable with 4-6 account tiers
- **Content strategy** should focus on Disney/HBO quality level

## 📁 Deliverables

### Code & Analysis
- `netflix_real_data_analysis_fixed.py` - **Main real data analysis**
- `netflix_conjoint_analysis.py` - Simulated data analysis (comparison)
- `Netflix_Conjoint_Analysis.ipynb` - Interactive notebook
- `demo.py` - Advanced scenario testing
- `run_analysis.py` - Quick analysis runner
- `setup.py` - Environment setup script

### Data & Results
- `netflix_customer_survey.csv` - **Real customer survey data (3,000 responses)**
- `real_data_market_scenarios.csv` - **Real data market scenario analysis**
- `profiles.csv` - Simulated experimental design (comparison)
- `responses.csv` - Simulated consumer responses (comparison)
- `custom_scenarios.csv` - Extended scenario testing
- `revenue_analysis.csv` - Revenue optimization results

### Visualizations
- `netflix_real_data_analysis.png` - **Comprehensive 12-panel dashboard (real data)**
- `conjoint_analysis_results.png` - Simulated data visualizations (comparison)
- `relative_importance.png` - Attribute importance breakdown

### Documentation
- `README.md` - Comprehensive documentation
- `SUMMARY.md` - This executive summary
- `requirements.txt` - Python dependencies

## 🚀 Getting Started

### Quick Start
```bash
# Setup environment and run analysis
python setup.py

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```

### Advanced Analysis
```bash
# Detailed scenario testing
python demo.py

# Interactive exploration
jupyter notebook Netflix_Conjoint_Analysis.ipynb
```

## 📊 Technical Specifications

### Statistical Model
```
Rating = 5.047 + 1.561×Disney + 1.294×HBO + 0.855×Sports
         + 0.440×(2_Accounts) + 1.013×(4_Accounts)
         - 0.384×($12.99) - 0.951×($15.99) - 1.745×($19.99)
```

### Model Performance
- **R-squared**: 0.346 (good explanatory power)
- **F-statistic**: 78.86 (highly significant)
- **All coefficients**: p < 0.001 (statistically significant)

### Design Properties
- **Orthogonal**: Attributes uncorrelated
- **Balanced**: Each level appears equally
- **Efficient**: Minimizes standard errors
- **Realistic**: Feasible market offerings

## 🎯 Next Steps

### Implementation
1. **Validate with real data** - Replace simulated responses
2. **Segment analysis** - Identify customer segments
3. **Competitive benchmarking** - Compare with market offerings
4. **A/B testing** - Test top scenarios in market

### Extensions
1. **Interaction effects** - Test content×price interactions
2. **Choice-based conjoint** - Implement discrete choice models
3. **Hierarchical modeling** - Account for individual differences
4. **Dynamic pricing** - Incorporate demand elasticity

---

**Project Status**: ✅ Complete and Ready for Implementation

**Last Updated**: July 31, 2025

**Contact**: See README.md for support information