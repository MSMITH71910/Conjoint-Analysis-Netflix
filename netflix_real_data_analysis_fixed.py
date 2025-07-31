#!/usr/bin/env python3
"""
Netflix Conjoint Analysis - Real Data Implementation (Fixed)
===========================================================

This module implements a comprehensive conjoint analysis using real Netflix customer survey data.
The analysis addresses Netflix's key business challenges:
1. Increase subscriptions
2. Optimize pricing strategy
3. Add revenue streams
4. Expand to new markets

Author: MSMITH71910
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class NetflixRealDataAnalysis:
    """
    Netflix Conjoint Analysis using real customer survey data.
    
    This class implements a complete conjoint analysis workflow including:
    - Data preprocessing and cleaning
    - Dummy variable creation
    - Statistical modeling (OLS and Logistic Regression)
    - Part-worth utility calculation
    - Market simulation and scenario analysis
    - Business insights generation
    """
    
    def __init__(self, data_file='netflix_customer_survey.csv'):
        """
        Initialize the analysis with real Netflix survey data.
        
        Args:
            data_file (str): Path to the Netflix customer survey CSV file
        """
        self.data_file = data_file
        self.df = None
        self.X = None
        self.y = None
        self.X_encoded = None
        self.model_ols = None
        self.model_logit = None
        self.part_worths = {}
        self.relative_importance = {}
        
        print("Netflix Real Data Conjoint Analysis")
        print("=" * 50)
        print("Loading and analyzing real customer survey data...")
        
    def load_and_explore_data(self):
        """Load the data and perform initial exploration."""
        print("\nStep 1: Loading and exploring data...")
        
        # Load data
        self.df = pd.read_csv(self.data_file)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of customers: {self.df['customerid'].nunique()}")
        print(f"Total responses: {len(self.df)}")
        
        # Display basic info
        print("\nData structure:")
        print(self.df.head())
        
        print("\nAttribute levels:")
        for col in ['NumberAccounts', 'price', 'ExtraContent', 'ads']:
            print(f"  {col}: {sorted(self.df[col].unique())}")
        
        print(f"\nSelection rate: {self.df['selected'].mean():.3f}")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data for analysis."""
        print("\nStep 2: Preprocessing data...")
        
        # Separate features and target
        self.y = self.df['selected'].astype(float)
        self.X = self.df.drop(columns=['selected', 'customerid'])
        
        # Ensure numeric columns are float
        self.X['NumberAccounts'] = self.X['NumberAccounts'].astype(float)
        self.X['price'] = self.X['price'].astype(float)
        
        # Create dummy variables for categorical variables (drop first to avoid multicollinearity)
        self.X_encoded = pd.get_dummies(self.X, columns=['ExtraContent', 'ads'], 
                                       prefix=['Content', 'Ads'], drop_first=True)
        
        # Ensure all columns are float
        self.X_encoded = self.X_encoded.astype(float)
        
        print("Dummy variables created:")
        print(self.X_encoded.columns.tolist())
        
        # Display correlation matrix
        print("\nFeature correlations with selection:")
        correlations = pd.concat([self.X_encoded, self.y], axis=1).corr()['selected'].sort_values(ascending=False)
        print(correlations.drop('selected'))
        
        return self.X_encoded, self.y
    
    def fit_models(self):
        """Fit both OLS and Logistic Regression models."""
        print("\nStep 3: Fitting statistical models...")
        
        # Add constant for OLS
        X_with_const = sm.add_constant(self.X_encoded)
        
        # Fit OLS model (for part-worth utilities)
        print("Fitting OLS model...")
        self.model_ols = sm.OLS(self.y, X_with_const).fit()
        print("OLS Model Summary:")
        print(self.model_ols.summary())
        
        # Fit Logistic Regression model (for choice prediction)
        print("\nFitting Logistic Regression model...")
        self.model_logit = sm.Logit(self.y, X_with_const).fit()
        print("Logistic Regression Summary:")
        print(self.model_logit.summary())
        
        return self.model_ols, self.model_logit
    
    def calculate_part_worths(self):
        """Calculate part-worth utilities from the OLS model."""
        print("\nStep 4: Calculating part-worth utilities...")
        
        coefficients = self.model_ols.params
        
        # Extract part-worths for each attribute
        self.part_worths = {
            'Intercept': coefficients['const'],
            'NumberAccounts': coefficients['NumberAccounts'],
            'Price': coefficients['price'],
            'Content': {},
            'Ads': {}
        }
        
        # Content part-worths (relative to baseline - Disney is the dropped category)
        content_cols = [col for col in coefficients.index if col.startswith('Content_')]
        baseline_content = 'Disney'  # This was dropped, so it's the baseline
        self.part_worths['Content'][baseline_content] = 0.0  # Baseline
        
        for col in content_cols:
            content_type = col.replace('Content_', '')
            self.part_worths['Content'][content_type] = coefficients[col]
        
        # Ads part-worths (relative to baseline - 'none' is the dropped category)
        ads_cols = [col for col in coefficients.index if col.startswith('Ads_')]
        baseline_ads = 'none'  # This was dropped, so it's the baseline
        self.part_worths['Ads'][baseline_ads] = 0.0  # Baseline
        
        for col in ads_cols:
            ads_type = col.replace('Ads_', '')
            self.part_worths['Ads'][ads_type] = coefficients[col]
        
        # Calculate relative importance
        ranges = {}
        ranges['NumberAccounts'] = abs(self.part_worths['NumberAccounts'] * 
                                     (self.X['NumberAccounts'].max() - self.X['NumberAccounts'].min()))
        ranges['Price'] = abs(self.part_worths['Price'] * 
                            (self.X['price'].max() - self.X['price'].min()))
        
        if self.part_worths['Content']:
            content_values = list(self.part_worths['Content'].values())
            ranges['Content'] = max(content_values) - min(content_values)
        
        if self.part_worths['Ads']:
            ads_values = list(self.part_worths['Ads'].values())
            ranges['Ads'] = max(ads_values) - min(ads_values)
        
        total_range = sum(ranges.values())
        self.relative_importance = {attr: (range_val / total_range) * 100 
                                  for attr, range_val in ranges.items()}
        
        # Display results
        print("\nPart-Worth Utilities:")
        print(f"Intercept: {self.part_worths['Intercept']:.4f}")
        print(f"NumberAccounts: {self.part_worths['NumberAccounts']:.4f}")
        print(f"Price: {self.part_worths['Price']:.4f}")
        
        print("\nContent Utilities (relative to Disney baseline):")
        for content, utility in self.part_worths['Content'].items():
            print(f"  {content}: {utility:.4f}")
        
        print("\nAds Utilities (relative to 'none' baseline):")
        for ads, utility in self.part_worths['Ads'].items():
            print(f"  {ads}: {utility:.4f}")
        
        print("\nRelative Importance:")
        for attr, importance in self.relative_importance.items():
            print(f"  {attr}: {importance:.1f}%")
        
        return self.part_worths, self.relative_importance
    
    def predict_scenario_utility(self, accounts, price, content, ads):
        """Predict utility for a given scenario using the OLS model."""
        utility = self.part_worths['Intercept']
        utility += self.part_worths['NumberAccounts'] * accounts
        utility += self.part_worths['Price'] * price
        utility += self.part_worths['Content'].get(content, 0)
        utility += self.part_worths['Ads'].get(ads, 0)
        return utility
    
    def simulate_market_scenarios(self):
        """Simulate different market scenarios."""
        print("\nStep 5: Simulating market scenarios...")
        
        # Define scenarios to test
        scenarios = [
            {'name': 'Current Basic', 'NumberAccounts': 1, 'price': 8, 'ExtraContent': 'less content', 'ads': 'none'},
            {'name': 'Disney Premium', 'NumberAccounts': 4, 'price': 15, 'ExtraContent': 'Disney', 'ads': 'none'},
            {'name': 'HBO Standard', 'NumberAccounts': 2, 'price': 12, 'ExtraContent': 'HBO', 'ads': 'one_per_show'},
            {'name': 'Marvel Family', 'NumberAccounts': 6, 'price': 18, 'ExtraContent': 'Marvel', 'ads': 'one_per_day'},
            {'name': 'Sports Package', 'NumberAccounts': 4, 'price': 20, 'ExtraContent': 'Soccer', 'ads': 'none'},
            {'name': 'Prime Competitor', 'NumberAccounts': 3, 'price': 10, 'ExtraContent': 'Prime originals', 'ads': 'one_per_show'},
            {'name': 'Budget Option', 'NumberAccounts': 1, 'price': 8, 'ExtraContent': 'less content', 'ads': 'one_per_day'},
            {'name': 'Premium All', 'NumberAccounts': 6, 'price': 20, 'ExtraContent': 'HBO', 'ads': 'none'}
        ]
        
        results = []
        
        for scenario in scenarios:
            # Calculate utility using our prediction function
            utility = self.predict_scenario_utility(
                scenario['NumberAccounts'], 
                scenario['price'], 
                scenario['ExtraContent'], 
                scenario['ads']
            )
            
            # Convert utility to probability (sigmoid transformation)
            prob = 1 / (1 + np.exp(-utility))
            
            results.append({
                'Scenario': scenario['name'],
                'Accounts': scenario['NumberAccounts'],
                'Price': f"${scenario['price']}",
                'Content': scenario['ExtraContent'],
                'Ads': scenario['ads'],
                'Utility': utility,
                'Choice_Probability': prob,
                'Appeal_Score': min(10, max(1, round(prob * 10)))  # Scale to 1-10
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Choice_Probability', ascending=False)
        
        print("\nMarket Scenario Analysis:")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\nStep 6: Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Selection rate by content type
        plt.subplot(3, 4, 1)
        content_selection = self.df.groupby('ExtraContent')['selected'].mean().sort_values(ascending=False)
        content_selection.plot(kind='bar', color='skyblue')
        plt.title('Selection Rate by Content Type', fontsize=12, fontweight='bold')
        plt.xlabel('Content Type')
        plt.ylabel('Selection Rate')
        plt.xticks(rotation=45)
        
        # 2. Selection rate by price
        plt.subplot(3, 4, 2)
        price_selection = self.df.groupby('price')['selected'].mean().sort_index()
        price_selection.plot(kind='line', marker='o', color='green')
        plt.title('Selection Rate by Price', fontsize=12, fontweight='bold')
        plt.xlabel('Price ($)')
        plt.ylabel('Selection Rate')
        
        # 3. Selection rate by number of accounts
        plt.subplot(3, 4, 3)
        accounts_selection = self.df.groupby('NumberAccounts')['selected'].mean().sort_index()
        accounts_selection.plot(kind='bar', color='orange')
        plt.title('Selection Rate by Number of Accounts', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Accounts')
        plt.ylabel('Selection Rate')
        
        # 4. Selection rate by ads
        plt.subplot(3, 4, 4)
        ads_selection = self.df.groupby('ads')['selected'].mean().sort_values(ascending=False)
        ads_selection.plot(kind='bar', color='red')
        plt.title('Selection Rate by Ad Frequency', fontsize=12, fontweight='bold')
        plt.xlabel('Ad Frequency')
        plt.ylabel('Selection Rate')
        plt.xticks(rotation=45)
        
        # 5. Part-worth utilities for content
        plt.subplot(3, 4, 5)
        if self.part_worths['Content']:
            content_utils = pd.Series(self.part_worths['Content'])
            content_utils.plot(kind='bar', color='purple')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title('Content Part-Worth Utilities', fontsize=12, fontweight='bold')
            plt.xlabel('Content Type')
            plt.ylabel('Utility')
            plt.xticks(rotation=45)
        
        # 6. Part-worth utilities for ads
        plt.subplot(3, 4, 6)
        if self.part_worths['Ads']:
            ads_utils = pd.Series(self.part_worths['Ads'])
            ads_utils.plot(kind='bar', color='brown')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title('Ads Part-Worth Utilities', fontsize=12, fontweight='bold')
            plt.xlabel('Ad Frequency')
            plt.ylabel('Utility')
            plt.xticks(rotation=45)
        
        # 7. Relative importance pie chart
        plt.subplot(3, 4, 7)
        importance_data = pd.Series(self.relative_importance)
        plt.pie(importance_data.values, labels=importance_data.index, autopct='%1.1f%%')
        plt.title('Relative Attribute Importance', fontsize=12, fontweight='bold')
        
        # 8. Price sensitivity analysis
        plt.subplot(3, 4, 8)
        price_coeff = self.part_worths['Price']
        prices = range(8, 25, 2)
        utilities = [price_coeff * (p - 8) for p in prices]  # Relative to $8 baseline
        plt.plot(prices, utilities, marker='o', color='red')
        plt.title('Price Sensitivity Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Price ($)')
        plt.ylabel('Utility Change')
        plt.grid(True, alpha=0.3)
        
        # 9. Customer distribution by selection
        plt.subplot(3, 4, 9)
        selection_dist = self.df['selected'].value_counts()
        plt.pie(selection_dist.values, labels=['Not Selected', 'Selected'], autopct='%1.1f%%')
        plt.title('Overall Selection Distribution', fontsize=12, fontweight='bold')
        
        # 10. Accounts vs Price heatmap
        plt.subplot(3, 4, 10)
        pivot_data = self.df.pivot_table(values='selected', index='NumberAccounts', 
                                       columns='price', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r')
        plt.title('Selection Rate: Accounts vs Price', fontsize=12, fontweight='bold')
        
        # 11. Content vs Ads heatmap
        plt.subplot(3, 4, 11)
        pivot_content_ads = self.df.pivot_table(values='selected', index='ExtraContent', 
                                              columns='ads', aggfunc='mean')
        sns.heatmap(pivot_content_ads, annot=True, fmt='.2f', cmap='RdYlBu_r')
        plt.title('Selection Rate: Content vs Ads', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 12. Model performance metrics
        plt.subplot(3, 4, 12)
        y_pred_ols = (self.model_ols.predict(sm.add_constant(self.X_encoded)) > 0.5).astype(int)
        
        accuracy_ols = accuracy_score(self.y, y_pred_ols)
        r_squared = self.model_ols.rsquared
        
        metrics = ['Accuracy', 'R-squared']
        values = [accuracy_ols, r_squared]
        
        plt.bar(metrics, values, color=['blue', 'green'])
        plt.title('Model Performance Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        for i, val in enumerate(values):
            plt.text(i, val + 0.01, f'{val:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('netflix_real_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'netflix_real_data_analysis.png'")
    
    def generate_business_insights(self):
        """Generate actionable business insights."""
        print("\nStep 7: Business Insights and Recommendations")
        print("=" * 60)
        
        # Content insights
        if self.part_worths['Content']:
            best_content = max(self.part_worths['Content'].items(), key=lambda x: x[1])
            worst_content = min(self.part_worths['Content'].items(), key=lambda x: x[1])
            
            print("1. CONTENT STRATEGY:")
            print(f"   • Best performing content: {best_content[0]} (utility: {best_content[1]:.3f})")
            print(f"   • Worst performing content: {worst_content[0]} (utility: {worst_content[1]:.3f})")
            print(f"   • Content importance: {self.relative_importance.get('Content', 0):.1f}% of decision")
        
        # Pricing insights
        price_coeff = self.part_worths['Price']
        print(f"\n2. PRICING STRATEGY:")
        print(f"   • Price sensitivity coefficient: {price_coeff:.4f}")
        if price_coeff < 0:
            print("   • Customers are price-sensitive (negative coefficient)")
            print("   • Each $1 increase reduces utility by {:.3f}".format(abs(price_coeff)))
        else:
            print("   • Customers associate higher price with higher value")
        print(f"   • Price importance: {self.relative_importance.get('Price', 0):.1f}% of decision")
        
        # Accounts insights
        accounts_coeff = self.part_worths['NumberAccounts']
        print(f"\n3. ACCOUNTS STRATEGY:")
        print(f"   • Accounts coefficient: {accounts_coeff:.4f}")
        if accounts_coeff > 0:
            print("   • Customers value additional accounts")
            print("   • Each additional account increases utility by {:.3f}".format(accounts_coeff))
        print(f"   • Accounts importance: {self.relative_importance.get('NumberAccounts', 0):.1f}% of decision")
        
        # Ads insights
        if self.part_worths['Ads']:
            best_ads = max(self.part_worths['Ads'].items(), key=lambda x: x[1])
            worst_ads = min(self.part_worths['Ads'].items(), key=lambda x: x[1])
            
            print(f"\n4. ADVERTISING STRATEGY:")
            print(f"   • Most acceptable ads: {best_ads[0]} (utility: {best_ads[1]:.3f})")
            print(f"   • Least acceptable ads: {worst_ads[0]} (utility: {worst_ads[1]:.3f})")
            print(f"   • Ads importance: {self.relative_importance.get('Ads', 0):.1f}% of decision")
        
        # Business problem solutions
        print(f"\n5. BUSINESS PROBLEM SOLUTIONS:")
        print("   • INCREASE SUBSCRIPTIONS:")
        if self.part_worths['Content']:
            top_content = max(self.part_worths['Content'].items(), key=lambda x: x[1])[0]
            print(f"     - Focus on {top_content} content")
        if accounts_coeff > 0:
            print("     - Promote multi-account plans")
        
        print("   • OPTIMIZE PRICING:")
        if price_coeff < 0:
            print("     - Be cautious with price increases")
            print("     - Bundle value-added features with price increases")
        
        print("   • ADD REVENUE STREAMS:")
        if self.part_worths['Content']:
            print("     - Premium content tiers")
        if self.part_worths['Ads']:
            print("     - Ad-supported tiers for price-sensitive customers")
        
        print("   • NEW MARKETS:")
        print("     - Replicate successful attribute combinations")
        print("     - Adapt content mix to local preferences")
    
    def run_complete_analysis(self):
        """Run the complete analysis workflow."""
        try:
            # Load and explore data
            self.load_and_explore_data()
            
            # Preprocess data
            self.preprocess_data()
            
            # Fit models
            self.fit_models()
            
            # Calculate part-worths
            self.calculate_part_worths()
            
            # Simulate scenarios
            scenarios_df = self.simulate_market_scenarios()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate insights
            self.generate_business_insights()
            
            # Save results
            scenarios_df.to_csv('real_data_market_scenarios.csv', index=False)
            
            print(f"\n" + "=" * 60)
            print("Analysis Complete!")
            print("=" * 60)
            print("Generated files:")
            print("• netflix_real_data_analysis.png - Comprehensive visualizations")
            print("• real_data_market_scenarios.csv - Market scenario analysis")
            
            return {
                'model_ols': self.model_ols,
                'model_logit': self.model_logit,
                'part_worths': self.part_worths,
                'relative_importance': self.relative_importance,
                'scenarios': scenarios_df
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            raise

def main():
    """Main function to run the analysis."""
    # Initialize and run analysis
    netflix_analysis = NetflixRealDataAnalysis()
    results = netflix_analysis.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()