"""
Netflix Conjoint Analysis
========================

This script implements a comprehensive conjoint analysis for Netflix to address:
1. Increase the number of subscriptions
2. Increase the price (without losing too many subscribers)
3. Add revenue streams
4. Add new markets

Attributes and Levels:
- Content: [Netflix Originals, Disney, HBO, Sports]
- Number of accounts (simultaneous streams): [1, 2, 4]
- Price: [$8.99, $12.99, $15.99, $19.99]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class NetflixConjointAnalysis:
    def __init__(self):
        """Initialize the conjoint analysis with attributes and levels."""
        # Define attributes and levels
        self.attributes = {
            'Content': ['Netflix Originals', 'Disney', 'HBO', 'Sports'],
            'Accounts': [1, 2, 4],
            'Price': [8.99, 12.99, 15.99, 19.99]
        }
        
        # Base levels for dummy coding
        self.base_levels = {
            'Content': 'Netflix Originals',
            'Accounts': 1,
            'Price': 8.99
        }
        
        self.design_matrix = None
        self.profiles = None
        self.responses = None
        self.model_results = None
        self.part_worths = None
        
    def create_fractional_factorial_design(self):
        """Create a balanced fractional factorial design with 12 profiles."""
        print("Step 1: Creating fractional factorial design...")
        
        # Create 12 profiles using a balanced design
        profiles = []
        
        # Design matrix ensuring balance and orthogonality
        design_plan = [
            (0, 0, 0),  # Netflix, 1 account, $8.99
            (0, 1, 1),  # Netflix, 2 accounts, $12.99
            (0, 2, 2),  # Netflix, 4 accounts, $15.99
            (1, 0, 3),  # Disney, 1 account, $19.99
            (1, 1, 0),  # Disney, 2 accounts, $8.99
            (1, 2, 1),  # Disney, 4 accounts, $12.99
            (2, 0, 2),  # HBO, 1 account, $15.99
            (2, 1, 3),  # HBO, 2 accounts, $19.99
            (2, 2, 0),  # HBO, 4 accounts, $8.99
            (3, 0, 1),  # Sports, 1 account, $12.99
            (3, 1, 2),  # Sports, 2 accounts, $15.99
            (3, 2, 3),  # Sports, 4 accounts, $19.99
        ]
        
        for i, (content_idx, account_idx, price_idx) in enumerate(design_plan):
            profile = {
                'Profile_ID': i + 1,
                'Content': self.attributes['Content'][content_idx],
                'Accounts': self.attributes['Accounts'][account_idx],
                'Price': self.attributes['Price'][price_idx]
            }
            profiles.append(profile)
        
        self.profiles = pd.DataFrame(profiles)
        
        # Check balance
        print("\nDesign Balance Check:")
        for attr in ['Content', 'Accounts', 'Price']:
            print(f"{attr}: {self.profiles[attr].value_counts().to_dict()}")
        
        return self.profiles
    
    def create_design_matrix(self):
        """Create dummy-coded design matrix for regression analysis."""
        print("\nStep 2: Creating design matrix with dummy coding...")
        
        design_data = []
        
        for _, profile in self.profiles.iterrows():
            row = {'Profile_ID': profile['Profile_ID']}
            
            # Content dummies (Netflix Originals as base)
            for content in ['Disney', 'HBO', 'Sports']:
                row[f'Content_{content}'] = 1 if profile['Content'] == content else 0
            
            # Accounts dummies (1 account as base)
            for accounts in [2, 4]:
                row[f'Accounts_{accounts}'] = 1 if profile['Accounts'] == accounts else 0
            
            # Price dummies ($8.99 as base)
            for price in [12.99, 15.99, 19.99]:
                row[f'Price_{price}'] = 1 if profile['Price'] == price else 0
            
            design_data.append(row)
        
        self.design_matrix = pd.DataFrame(design_data)
        return self.design_matrix
    
    def simulate_responses(self, n_respondents=100):
        """Simulate responses from n_respondents using a utility function."""
        print(f"\nStep 3: Simulating responses from {n_respondents} respondents...")
        
        # Define true utility coefficients (these represent market preferences)
        true_coefficients = {
            'Intercept': 5.0,
            'Content_Disney': 1.5,
            'Content_HBO': 1.2,
            'Content_Sports': 0.8,
            'Accounts_2': 0.5,
            'Accounts_4': 1.0,
            'Price_12.99': -0.5,
            'Price_15.99': -1.0,
            'Price_19.99': -1.8
        }
        
        responses_data = []
        
        for respondent_id in range(1, n_respondents + 1):
            # Add individual heterogeneity
            individual_coeffs = {}
            for coeff, value in true_coefficients.items():
                if coeff == 'Intercept':
                    individual_coeffs[coeff] = value + np.random.normal(0, 0.5)
                else:
                    individual_coeffs[coeff] = value + np.random.normal(0, 0.3)
            
            for _, profile in self.profiles.iterrows():
                # Calculate utility
                utility = individual_coeffs['Intercept']
                
                # Content effects
                if profile['Content'] == 'Disney':
                    utility += individual_coeffs['Content_Disney']
                elif profile['Content'] == 'HBO':
                    utility += individual_coeffs['Content_HBO']
                elif profile['Content'] == 'Sports':
                    utility += individual_coeffs['Content_Sports']
                
                # Accounts effects
                if profile['Accounts'] == 2:
                    utility += individual_coeffs['Accounts_2']
                elif profile['Accounts'] == 4:
                    utility += individual_coeffs['Accounts_4']
                
                # Price effects
                if profile['Price'] == 12.99:
                    utility += individual_coeffs['Price_12.99']
                elif profile['Price'] == 15.99:
                    utility += individual_coeffs['Price_15.99']
                elif profile['Price'] == 19.99:
                    utility += individual_coeffs['Price_19.99']
                
                # Add random error and convert to 1-10 scale
                rating = utility + np.random.normal(0, 1)
                rating = max(1, min(10, round(rating)))
                
                responses_data.append({
                    'Respondent_ID': respondent_id,
                    'Profile_ID': profile['Profile_ID'],
                    'Content': profile['Content'],
                    'Accounts': profile['Accounts'],
                    'Price': profile['Price'],
                    'Rating': rating
                })
        
        self.responses = pd.DataFrame(responses_data)
        print(f"Generated {len(self.responses)} responses")
        return self.responses
    
    def estimate_model(self):
        """Estimate the conjoint model using OLS regression."""
        print("\nStep 4: Estimating conjoint model...")
        
        # Merge responses with design matrix
        model_data = self.responses.merge(self.design_matrix, on='Profile_ID')
        
        # Prepare variables for regression
        X_columns = [col for col in self.design_matrix.columns if col != 'Profile_ID']
        X = model_data[X_columns]
        X = sm.add_constant(X)  # Add intercept
        y = model_data['Rating']
        
        # Fit OLS model
        self.model_results = sm.OLS(y, X).fit()
        
        print("Model Summary:")
        print(self.model_results.summary())
        
        return self.model_results
    
    def calculate_part_worths(self):
        """Calculate part-worth utilities and relative importance."""
        print("\nStep 5: Calculating part-worth utilities...")
        
        coefficients = self.model_results.params
        
        # Part-worth utilities
        part_worths = {}
        
        # Content part-worths
        part_worths['Content'] = {
            'Netflix Originals': 0,  # Base level
            'Disney': coefficients.get('Content_Disney', 0),
            'HBO': coefficients.get('Content_HBO', 0),
            'Sports': coefficients.get('Content_Sports', 0)
        }
        
        # Accounts part-worths
        part_worths['Accounts'] = {
            1: 0,  # Base level
            2: coefficients.get('Accounts_2', 0),
            4: coefficients.get('Accounts_4', 0)
        }
        
        # Price part-worths
        part_worths['Price'] = {
            8.99: 0,  # Base level
            12.99: coefficients.get('Price_12.99', 0),
            15.99: coefficients.get('Price_15.99', 0),
            19.99: coefficients.get('Price_19.99', 0)
        }
        
        self.part_worths = part_worths
        
        # Calculate relative importance
        ranges = {}
        for attribute, utilities in part_worths.items():
            ranges[attribute] = max(utilities.values()) - min(utilities.values())
        
        total_range = sum(ranges.values())
        relative_importance = {attr: (range_val / total_range) * 100 
                             for attr, range_val in ranges.items()}
        
        print("\nPart-Worth Utilities:")
        for attribute, utilities in part_worths.items():
            print(f"\n{attribute}:")
            for level, utility in utilities.items():
                print(f"  {level}: {utility:.3f}")
        
        print("\nRelative Importance:")
        for attribute, importance in relative_importance.items():
            print(f"  {attribute}: {importance:.1f}%")
        
        return part_worths, relative_importance
    
    def simulate_market_scenarios(self):
        """Simulate different market scenarios to answer business questions."""
        print("\nStep 6: Simulating market scenarios...")
        
        scenarios = {
            'Current Base': {'Content': 'Netflix Originals', 'Accounts': 1, 'Price': 8.99},
            'Premium Content': {'Content': 'Disney', 'Accounts': 2, 'Price': 12.99},
            'Sports Package': {'Content': 'Sports', 'Accounts': 2, 'Price': 15.99},
            'Family Plan': {'Content': 'Netflix Originals', 'Accounts': 4, 'Price': 15.99},
            'Premium All': {'Content': 'HBO', 'Accounts': 4, 'Price': 19.99}
        }
        
        scenario_utilities = {}
        
        for scenario_name, attributes in scenarios.items():
            utility = self.model_results.params['const']  # Intercept
            
            # Add content utility
            if attributes['Content'] != 'Netflix Originals':
                content_coeff = f"Content_{attributes['Content']}"
                utility += self.model_results.params.get(content_coeff, 0)
            
            # Add accounts utility
            if attributes['Accounts'] != 1:
                accounts_coeff = f"Accounts_{attributes['Accounts']}"
                utility += self.model_results.params.get(accounts_coeff, 0)
            
            # Add price utility
            if attributes['Price'] != 8.99:
                price_coeff = f"Price_{attributes['Price']}"
                utility += self.model_results.params.get(price_coeff, 0)
            
            scenario_utilities[scenario_name] = utility
        
        # Convert utilities to market shares using logit model
        exp_utilities = {name: np.exp(utility) for name, utility in scenario_utilities.items()}
        total_exp_utility = sum(exp_utilities.values())
        market_shares = {name: (exp_util / total_exp_utility) * 100 
                        for name, exp_util in exp_utilities.items()}
        
        print("\nMarket Scenario Analysis:")
        print("=" * 50)
        
        scenario_df = pd.DataFrame([
            {
                'Scenario': name,
                'Content': attrs['Content'],
                'Accounts': attrs['Accounts'],
                'Price': f"${attrs['Price']}",
                'Utility': scenario_utilities[name],
                'Market Share (%)': market_shares[name],
                'Revenue Index': market_shares[name] * attrs['Price'] / 100
            }
            for name, attrs in scenarios.items()
        ])
        
        print(scenario_df.to_string(index=False))
        
        return scenario_df
    
    def create_visualizations(self):
        """Create visualizations for the conjoint analysis results."""
        print("\nStep 7: Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Netflix Conjoint Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Part-worth utilities plot
        ax1 = axes[0, 0]
        content_utils = list(self.part_worths['Content'].values())
        content_labels = list(self.part_worths['Content'].keys())
        
        bars1 = ax1.bar(content_labels, content_utils, color=['#E50914', '#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Content Part-Worth Utilities')
        ax1.set_ylabel('Utility')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, content_utils):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 2. Accounts part-worth utilities
        ax2 = axes[0, 1]
        accounts_utils = list(self.part_worths['Accounts'].values())
        accounts_labels = [f'{k} Account{"s" if k > 1 else ""}' for k in self.part_worths['Accounts'].keys()]
        
        bars2 = ax2.bar(accounts_labels, accounts_utils, color=['#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('Accounts Part-Worth Utilities')
        ax2.set_ylabel('Utility')
        
        for bar, value in zip(bars2, accounts_utils):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 3. Price part-worth utilities
        ax3 = axes[1, 0]
        price_utils = list(self.part_worths['Price'].values())
        price_labels = [f'${k}' for k in self.part_worths['Price'].keys()]
        
        bars3 = ax3.bar(price_labels, price_utils, color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
        ax3.set_title('Price Part-Worth Utilities')
        ax3.set_ylabel('Utility')
        
        for bar, value in zip(bars3, price_utils):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 4. Response distribution
        ax4 = axes[1, 1]
        ax4.hist(self.responses['Rating'], bins=range(1, 12), alpha=0.7, color='#E50914', edgecolor='black')
        ax4.set_title('Distribution of Ratings')
        ax4.set_xlabel('Rating (1-10)')
        ax4.set_ylabel('Frequency')
        ax4.set_xticks(range(1, 11))
        
        plt.tight_layout()
        plt.savefig('/home/msmith/Conjoint-Analysis-Netflix/conjoint_analysis_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create relative importance pie chart
        fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Calculate relative importance
        ranges = {}
        for attribute, utilities in self.part_worths.items():
            ranges[attribute] = max(utilities.values()) - min(utilities.values())
        
        total_range = sum(ranges.values())
        relative_importance = {attr: (range_val / total_range) * 100 
                             for attr, range_val in ranges.items()}
        
        colors = ['#E50914', '#1f77b4', '#ff7f0e']
        wedges, texts, autotexts = ax.pie(relative_importance.values(), 
                                         labels=relative_importance.keys(),
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        
        ax.set_title('Relative Importance of Attributes', fontsize=14, fontweight='bold')
        
        plt.savefig('/home/msmith/Conjoint-Analysis-Netflix/relative_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_business_insights(self):
        """Generate business insights and recommendations."""
        print("\nStep 8: Business Insights and Recommendations")
        print("=" * 60)
        
        insights = []
        
        # Content insights
        content_utils = self.part_worths['Content']
        best_content = max(content_utils, key=content_utils.get)
        worst_content = min(content_utils, key=content_utils.get)
        
        insights.append(f"1. CONTENT STRATEGY:")
        insights.append(f"   • {best_content} content has the highest utility ({content_utils[best_content]:.3f})")
        insights.append(f"   • {worst_content} content has the lowest utility ({content_utils[worst_content]:.3f})")
        insights.append(f"   • Recommendation: Focus on acquiring {best_content} content")
        
        # Accounts insights
        accounts_utils = self.part_worths['Accounts']
        best_accounts = max(accounts_utils, key=accounts_utils.get)
        
        insights.append(f"\n2. ACCOUNT STRATEGY:")
        insights.append(f"   • {best_accounts} accounts has the highest utility ({accounts_utils[best_accounts]:.3f})")
        insights.append(f"   • Customers value multiple simultaneous streams")
        insights.append(f"   • Recommendation: Promote family/multi-user plans")
        
        # Price insights
        price_utils = self.part_worths['Price']
        price_sensitivity = abs(min(price_utils.values()))
        
        insights.append(f"\n3. PRICING STRATEGY:")
        insights.append(f"   • Price sensitivity is moderate (max negative utility: {min(price_utils.values()):.3f})")
        insights.append(f"   • $19.99 price point shows highest resistance")
        insights.append(f"   • Recommendation: Sweet spot appears to be $12.99-$15.99 range")
        
        # Business problem solutions
        insights.append(f"\n4. BUSINESS PROBLEM SOLUTIONS:")
        insights.append(f"   • INCREASE SUBSCRIPTIONS: Focus on {best_content} content + multi-account plans")
        insights.append(f"   • INCREASE PRICE: Gradual increases to $12.99-$15.99 with value additions")
        insights.append(f"   • ADD REVENUE STREAMS: Premium content tiers, sports packages")
        insights.append(f"   • NEW MARKETS: Replicate successful content + pricing strategy")
        
        for insight in insights:
            print(insight)
        
        return insights
    
    def run_complete_analysis(self):
        """Run the complete conjoint analysis pipeline."""
        print("Netflix Conjoint Analysis")
        print("=" * 50)
        
        # Step 1: Create design
        self.create_fractional_factorial_design()
        
        # Step 2: Create design matrix
        self.create_design_matrix()
        
        # Step 3: Simulate responses
        self.simulate_responses(n_respondents=100)
        
        # Step 4: Estimate model
        self.estimate_model()
        
        # Step 5: Calculate part-worths
        self.calculate_part_worths()
        
        # Step 6: Market scenarios
        scenario_results = self.simulate_market_scenarios()
        
        # Step 7: Create visualizations
        self.create_visualizations()
        
        # Step 8: Generate insights
        self.generate_business_insights()
        
        print("\nAnalysis complete! Check the generated visualizations and results.")
        
        return {
            'profiles': self.profiles,
            'responses': self.responses,
            'model_results': self.model_results,
            'part_worths': self.part_worths,
            'scenarios': scenario_results
        }

# Example usage
if __name__ == "__main__":
    # Initialize and run the analysis
    netflix_analysis = NetflixConjointAnalysis()
    results = netflix_analysis.run_complete_analysis()
    
    # Save results to CSV files
    results['profiles'].to_csv('/home/msmith/Conjoint-Analysis-Netflix/profiles.csv', index=False)
    results['responses'].to_csv('/home/msmith/Conjoint-Analysis-Netflix/responses.csv', index=False)
    results['scenarios'].to_csv('/home/msmith/Conjoint-Analysis-Netflix/market_scenarios.csv', index=False)
    
    print("\nResults saved to CSV files:")
    print("- profiles.csv: Experimental design profiles")
    print("- responses.csv: Simulated consumer responses")
    print("- market_scenarios.csv: Market scenario analysis")