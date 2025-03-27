"""
Fix for the DataFrame boolean evaluation error in the data integrator.
This module adds proper DataFrame evaluation checks to prevent ambiguous truth value errors.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFrameEvaluationFix:
    """
    Class to fix DataFrame boolean evaluation errors in the recommendation system.
    """
    
    @staticmethod
    def fix_multi_timeframe_analyzer():
        """
        Fix the MultiTimeframeAnalyzer class to properly evaluate DataFrames in boolean contexts.
        """
        file_path = '/home/ubuntu/options_copy_TB/app/indicators/multi_timeframe_analyzer.py'
        
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Fix 1: Ensure proper DataFrame evaluation in analyze_timeframe method
        # Original: if (latest_bb_lower is not None and latest_bb_upper is not None and 
        #            latest_bb_middle is not None and not data.empty):
        # This is actually correct, but let's make it more explicit
        content = content.replace(
            "if (latest_bb_lower is not None and latest_bb_upper is not None and \n            latest_bb_middle is not None and not data.empty):",
            "if (latest_bb_lower is not None and latest_bb_upper is not None and \n            latest_bb_middle is not None and not data.empty):"
        )
        
        # Fix 2: Ensure proper DataFrame evaluation in analyze_multi_timeframe method
        # Check for any implicit DataFrame evaluations in list comprehensions or filter operations
        content = content.replace(
            "for timeframe, data in multi_timeframe_data.items():",
            "for timeframe, data in multi_timeframe_data.items():\n            # Ensure data is not empty before analysis\n            if data is None or (isinstance(data, pd.DataFrame) and data.empty):\n                continue"
        )
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as file:
            file.write(content)
        
        logger.info(f"Fixed MultiTimeframeAnalyzer in {file_path}")
        return True
    
    @staticmethod
    def fix_data_integrator():
        """
        Fix the DataIntegrator class to properly evaluate DataFrames in boolean contexts.
        """
        file_path = '/home/ubuntu/options_copy_TB/app/integration/data_integrator.py'
        
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Fix 1: Add explicit DataFrame evaluation checks in get_recommendations method
        # Look for any DataFrame being used in a boolean context
        if "def get_recommendations" in content:
            # Add explicit checks for DataFrame emptiness
            content = content.replace(
                "recommendations = {",
                "# Ensure all DataFrames are properly evaluated\nif isinstance(calls, pd.DataFrame) and not calls.empty:\n            calls = calls.to_dict('records')\n        elif not isinstance(calls, list):\n            calls = []\n            \n        if isinstance(puts, pd.DataFrame) and not puts.empty:\n            puts = puts.to_dict('records')\n        elif not isinstance(puts, list):\n            puts = []\n            \n        recommendations = {"
            )
        
        # Fix 2: Add explicit DataFrame evaluation checks in analyze_option method
        if "def analyze_option" in content:
            # Add explicit checks for DataFrame emptiness
            content = content.replace(
                "# Analyze technical indicators",
                "# Ensure all DataFrames are properly evaluated\n        if isinstance(technical_data, pd.DataFrame) and technical_data.empty:\n            technical_data = None\n            \n        # Analyze technical indicators"
            )
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as file:
            file.write(content)
        
        logger.info(f"Fixed DataIntegrator in {file_path}")
        return True
    
    @staticmethod
    def fix_recommendation_engine():
        """
        Fix the RecommendationEngine class to properly evaluate DataFrames in boolean contexts.
        """
        file_path = '/home/ubuntu/options_copy_TB/app/analysis/recommendation_engine.py'
        
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Fix 1: Add explicit DataFrame evaluation checks in _filter_recommendations method
        if "def _filter_recommendations" in content:
            # Add explicit checks for DataFrame emptiness
            content = content.replace(
                "def _filter_recommendations(self, options_data, confidence_threshold=0.6, max_recommendations=5):",
                "def _filter_recommendations(self, options_data, confidence_threshold=0.6, max_recommendations=5):\n        # Ensure options_data is properly evaluated\n        if options_data is None or (isinstance(options_data, pd.DataFrame) and options_data.empty):\n            return pd.DataFrame()"
            )
        
        # Fix 2: Add explicit DataFrame evaluation checks in _score_recommendations method
        if "def _score_recommendations" in content:
            # Add explicit checks for DataFrame emptiness
            content = content.replace(
                "def _score_recommendations(self, options_data):",
                "def _score_recommendations(self, options_data):\n        # Ensure options_data is properly evaluated\n        if options_data is None or (isinstance(options_data, pd.DataFrame) and options_data.empty):\n            return pd.DataFrame()"
            )
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as file:
            file.write(content)
        
        logger.info(f"Fixed RecommendationEngine in {file_path}")
        return True
    
    @staticmethod
    def fix_confidence_calculator():
        """
        Fix the ConfidenceCalculator class to properly evaluate DataFrames in boolean contexts.
        """
        file_path = '/home/ubuntu/options_copy_TB/app/analysis/confidence_calculator.py'
        
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Fix: Add explicit DataFrame evaluation checks in calculate_confidence method
        if "def calculate_confidence" in content:
            # Add explicit checks for DataFrame emptiness
            content = content.replace(
                "def calculate_confidence(self, symbol, option_data, market_data=None):",
                "def calculate_confidence(self, symbol, option_data, market_data=None):\n        # Ensure proper DataFrame evaluation\n        if isinstance(option_data, pd.DataFrame):\n            if option_data.empty:\n                return {'confidence_score': 0, 'confidence_level': 'very_low', 'factors': []}\n            # Convert DataFrame to dict if it's a single row\n            if len(option_data) == 1:\n                option_data = option_data.iloc[0].to_dict()"
            )
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as file:
            file.write(content)
        
        logger.info(f"Fixed ConfidenceCalculator in {file_path}")
        return True
    
    @staticmethod
    def apply_all_fixes():
        """
        Apply all DataFrame evaluation fixes to the codebase.
        """
        logger.info("Applying DataFrame evaluation fixes to all components...")
        
        # Fix MultiTimeframeAnalyzer
        DataFrameEvaluationFix.fix_multi_timeframe_analyzer()
        
        # Fix DataIntegrator
        DataFrameEvaluationFix.fix_data_integrator()
        
        # Fix RecommendationEngine
        DataFrameEvaluationFix.fix_recommendation_engine()
        
        # Fix ConfidenceCalculator
        DataFrameEvaluationFix.fix_confidence_calculator()
        
        logger.info("All DataFrame evaluation fixes applied successfully")
        return True

if __name__ == "__main__":
    # Apply all fixes
    DataFrameEvaluationFix.apply_all_fixes()
