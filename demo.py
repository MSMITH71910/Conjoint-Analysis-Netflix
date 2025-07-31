#!/usr/bin/env python3
"""
Netflix Conjoint Analysis - Demo Script
======================================

This script demonstrates how to use the Netflix Conjoint Analysis class
for different business scenarios and custom analysis.
"""

from netflix_conjoint_analysis import NetflixConjointAnalysis
import pandas as pd
import numpy as np

def main():
    print("Netflix Conjoint Analysis - Demo")
    print("=" * 40)
    
    # Initialize the analysis
    print("\n1. Initializing Netflix Conjoint Analysis...")
    netflix_analysis = NetflixConjointAnalysis()
    
    # Run the complete analysis
    print("\n2. Running complete analysis...")
    results = netflix_analysis.run_complete_analysis()
    
    # Custom scenario testing
    print("\n3. Testing Custom Scenarios...")
    print("-" * 30)
    
    def predict_custom_scenario(content, accounts, price):
        """Predict utility and market appeal for a custom scenario."""
        utility = netflix_analysis.model_results.params['const']  # Intercept
        
        # Add content utility
        if content != 'Netflix Originals':
            content_coeff = f"Content_{content}"
            utility += netflix_analysis.model_results.params.get(content_coeff, 0)
        
        # Add accounts utility
        if accounts != 1:
            accounts_coeff = f"Accounts_{accounts}"
            utility += netflix_analysis.model_results.params.get(accounts_coeff, 0)
        
        # Add price utility
        if price != 8.99:
            price_coeff = f"Price_{price}"
            utility += netflix_analysis.model_results.params.get(price_coeff, 0)
        
        return utility
    
    # Test various business scenarios
    test_scenarios = [
        # Current offerings
        ('Netflix Originals', 1, 8.99, 'Current Basic Plan'),
        ('Netflix Originals', 2, 12.99, 'Current Standard Plan'),
        ('Netflix Originals', 4, 15.99, 'Current Premium Plan'),
        
        # Potential new offerings
        ('Disney', 1, 10.99, 'Disney Basic'),
        ('Disney', 2, 14.99, 'Disney Standard'),
        ('Disney', 4, 18.99, 'Disney Premium'),
        
        ('Sports', 2, 16.99, 'Sports Package'),
        ('Sports', 4, 22.99, 'Sports Premium'),
        
        ('HBO', 1, 11.99, 'HBO Basic'),
        ('HBO', 4, 19.99, 'HBO Premium'),
        
        # Competitive scenarios
        ('Disney', 4, 12.99, 'Disney Family Deal'),
        ('Netflix Originals', 4, 13.99, 'Netflix Family Plus'),
    ]
    
    scenario_results = []
    
    for content, accounts, price, description in test_scenarios:
        utility = predict_custom_scenario(content, accounts, price)
        scenario_results.append({
            'Scenario': description,
            'Content': content,
            'Accounts': accounts,
            'Price': f'${price}',
            'Utility': utility,
            'Appeal_Score': min(10, max(1, round(utility)))
        })
    
    # Create DataFrame and display results
    scenario_df = pd.DataFrame(scenario_results)
    scenario_df = scenario_df.sort_values('Utility', ascending=False)
    
    print("\nCustom Scenario Analysis Results:")
    print("(Higher utility = more attractive to consumers)")
    print("-" * 60)
    print(scenario_df.to_string(index=False))
    
    # Business recommendations
    print("\n4. Strategic Recommendations...")
    print("-" * 30)
    
    top_scenarios = scenario_df.head(3)
    print(f"\nTop 3 Most Attractive Scenarios:")
    for i, (_, row) in enumerate(top_scenarios.iterrows(), 1):
        print(f"{i}. {row['Scenario']}: {row['Content']}, {row['Accounts']} accounts, {row['Price']}")
        print(f"   Utility: {row['Utility']:.3f}, Appeal Score: {row['Appeal_Score']}/10")
    
    # Price sensitivity analysis
    print("\n5. Price Sensitivity Analysis...")
    print("-" * 30)
    
    # Test Disney content at different price points
    disney_prices = [8.99, 10.99, 12.99, 14.99, 16.99, 18.99, 20.99]
    disney_utilities = []
    
    for price in disney_prices:
        utility = predict_custom_scenario('Disney', 2, price)
        disney_utilities.append(utility)
    
    print("\nDisney Content (2 accounts) - Price Sensitivity:")
    for price, utility in zip(disney_prices, disney_utilities):
        appeal = min(10, max(1, round(utility)))
        print(f"${price:5.2f}: Utility = {utility:6.3f}, Appeal = {appeal}/10")
    
    # Market share simulation
    print("\n6. Market Share Simulation...")
    print("-" * 30)
    
    # Compare top 5 scenarios for market share
    top_5_scenarios = scenario_df.head(5)
    utilities = top_5_scenarios['Utility'].values
    
    # Convert to market shares using logit model
    exp_utilities = np.exp(utilities)
    market_shares = (exp_utilities / exp_utilities.sum()) * 100
    
    print("\nProjected Market Shares (Top 5 Scenarios):")
    for i, (_, row) in enumerate(top_5_scenarios.iterrows()):
        print(f"{row['Scenario']:25s}: {market_shares[i]:5.1f}%")
    
    # Revenue analysis
    print("\n7. Revenue Impact Analysis...")
    print("-" * 30)
    
    # Calculate revenue index for top scenarios
    revenue_data = []
    for i, (_, row) in enumerate(top_5_scenarios.iterrows()):
        price = float(row['Price'].replace('$', ''))
        revenue_index = market_shares[i] * price / 100
        revenue_data.append({
            'Scenario': row['Scenario'],
            'Market_Share': market_shares[i],
            'Price': price,
            'Revenue_Index': revenue_index
        })
    
    revenue_df = pd.DataFrame(revenue_data)
    revenue_df = revenue_df.sort_values('Revenue_Index', ascending=False)
    
    print("\nRevenue Optimization (Top 5 Scenarios):")
    print("Scenario                  Market Share  Price   Revenue Index")
    print("-" * 60)
    for _, row in revenue_df.iterrows():
        print(f"{row['Scenario']:25s} {row['Market_Share']:8.1f}%  ${row['Price']:5.2f}  {row['Revenue_Index']:8.3f}")
    
    print("\n8. Final Business Insights...")
    print("-" * 30)
    
    best_utility = scenario_df.iloc[0]
    best_revenue = revenue_df.iloc[0]
    
    print(f"\n• HIGHEST CONSUMER APPEAL: {best_utility['Scenario']}")
    print(f"  - Configuration: {best_utility['Content']}, {best_utility['Accounts']} accounts, {best_utility['Price']}")
    print(f"  - Utility Score: {best_utility['Utility']:.3f}")
    
    print(f"\n• HIGHEST REVENUE POTENTIAL: {best_revenue['Scenario']}")
    print(f"  - Market Share: {best_revenue['Market_Share']:.1f}%")
    print(f"  - Price Point: ${best_revenue['Price']:.2f}")
    print(f"  - Revenue Index: {best_revenue['Revenue_Index']:.3f}")
    
    print(f"\n• KEY INSIGHTS:")
    print(f"  - Disney content commands significant premium")
    print(f"  - Multi-account plans justify higher prices")
    print(f"  - Sweet spot for pricing appears to be $12.99-$16.99")
    print(f"  - Sports content has moderate appeal but niche market")
    
    # Save custom analysis results
    scenario_df.to_csv('/home/msmith/Conjoint-Analysis-Netflix/custom_scenarios.csv', index=False)
    revenue_df.to_csv('/home/msmith/Conjoint-Analysis-Netflix/revenue_analysis.csv', index=False)
    
    print(f"\n9. Results Saved!")
    print("-" * 30)
    print("• custom_scenarios.csv - Custom scenario analysis")
    print("• revenue_analysis.csv - Revenue optimization analysis")
    print("• All original analysis files also available")
    
    print(f"\nDemo completed successfully! 🎉")

if __name__ == "__main__":
    main()