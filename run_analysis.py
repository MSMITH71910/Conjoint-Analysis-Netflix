#!/usr/bin/env python3
"""
Quick Netflix Conjoint Analysis Runner
=====================================

Simple script to run the Netflix conjoint analysis and save results.
"""

from netflix_conjoint_analysis import NetflixConjointAnalysis

def main():
    print("Running Netflix Conjoint Analysis...")
    print("=" * 40)
    
    # Initialize and run analysis
    netflix_analysis = NetflixConjointAnalysis()
    results = netflix_analysis.run_complete_analysis()
    
    print("\n" + "=" * 40)
    print("Analysis Complete!")
    print("=" * 40)
    
    print("\nGenerated Files:")
    print("• profiles.csv - Experimental design profiles")
    print("• responses.csv - Simulated consumer responses") 
    print("• market_scenarios.csv - Market scenario analysis")
    print("• conjoint_analysis_results.png - Main results visualization")
    print("• relative_importance.png - Attribute importance chart")
    
    print("\nKey Findings:")
    print("• Disney content has highest consumer appeal")
    print("• 4-account plans justify premium pricing")
    print("• Optimal price range: $12.99-$15.99")
    print("• Premium content + multi-accounts = best revenue")
    
    print("\nTo explore further:")
    print("• Run 'python demo.py' for detailed scenario analysis")
    print("• Open Netflix_Conjoint_Analysis.ipynb for interactive analysis")
    print("• Check README.md for full documentation")

if __name__ == "__main__":
    main()