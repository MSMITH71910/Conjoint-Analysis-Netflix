#!/usr/bin/env python3
"""
Netflix Conjoint Analysis Setup Script
=====================================

This script sets up the environment and runs the Netflix conjoint analysis.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("Netflix Conjoint Analysis - Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('netflix_conjoint_analysis.py'):
        print("Error: Please run this script from the Conjoint-Analysis-Netflix directory")
        sys.exit(1)
    
    # Check if virtual environment exists
    if not os.path.exists('venv'):
        print("\n1. Creating virtual environment...")
        if not run_command('python3 -m venv venv', 'Virtual environment creation'):
            sys.exit(1)
    else:
        print("\n1. Virtual environment already exists ✓")
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not run_command('source venv/bin/activate && pip install -r requirements.txt', 
                      'Requirements installation'):
        sys.exit(1)
    
    # Run the analysis
    print("\n3. Running Netflix Conjoint Analysis...")
    if not run_command('source venv/bin/activate && python run_analysis.py', 
                      'Conjoint analysis execution'):
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("Setup Complete! 🎉")
    print("=" * 40)
    
    print("\nNext Steps:")
    print("1. Check the generated CSV files and PNG visualizations")
    print("2. Run 'source venv/bin/activate && python demo.py' for detailed analysis")
    print("3. Open Netflix_Conjoint_Analysis.ipynb in Jupyter for interactive exploration")
    print("4. Read README.md for comprehensive documentation")
    
    print("\nQuick Commands:")
    print("• Activate environment: source venv/bin/activate")
    print("• Run basic analysis: python run_analysis.py")
    print("• Run detailed demo: python demo.py")
    print("• Start Jupyter: jupyter notebook Netflix_Conjoint_Analysis.ipynb")

if __name__ == "__main__":
    main()