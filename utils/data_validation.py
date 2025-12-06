"""
Data validation utilities for basket prediction pipeline.

This module provides functions to validate data integrity and detect
common issues like missing user mappings for test baskets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)


def validate_data_files(data_path: Path) -> Dict[str, any]:
    """
    Validate basket prediction data files for consistency.
    
    Args:
        data_path: Path to data directory containing txt files
        
    Returns:
        Dictionary containing validation results and recommendations
    """
    validation_results = {
        'status': 'valid',
        'warnings': [],
        'errors': [],
        'recommendations': [],
        'statistics': {}
    }
    
    try:
        # Load data files
        train_u2b_file = data_path / "train_u2b.txt"
        train_b2i_file = data_path / "train_b2i.txt"
        test_b2i_file = data_path / "test_b2i.txt"
        test_u2b_file = data_path / "test_u2b.txt"
        
        # Check required files exist
        required_files = [train_u2b_file, train_b2i_file, test_b2i_file]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            validation_results['errors'].extend([f"Missing required file: {f}" for f in missing_files])
            validation_results['status'] = 'error'
            return validation_results
        
        # Load train user-basket mappings
        train_u2b = {}
        with open(train_u2b_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    user_id = int(parts[0])
                    basket_ids = [int(bid) for bid in parts[1:]]
                    train_u2b[user_id] = basket_ids
        
        # Load train basket-item mappings
        train_b2i = {}
        with open(train_b2i_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    basket_id = int(parts[0])
                    # Handle timestamp format (TGN)
                    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 8:
                        item_ids = [int(iid) for iid in parts[1:-1]]  # Exclude timestamp
                    else:
                        item_ids = [int(iid) for iid in parts[1:]]
                    train_b2i[basket_id] = item_ids
        
        # Load test basket-item mappings
        test_b2i = {}
        with open(test_b2i_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    basket_id = int(parts[0])
                    # Handle timestamp format (TGN)
                    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 8:
                        item_ids = [int(iid) for iid in parts[1:-1]]  # Exclude timestamp
                    else:
                        item_ids = [int(iid) for iid in parts[1:]]
                    test_b2i[basket_id] = item_ids
        
        # Load test user-basket mappings (optional)
        test_u2b = {}
        if test_u2b_file.exists():
            with open(test_u2b_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        user_id = int(parts[0])
                        test_basket_ids = [int(bid) for bid in parts[1:]]
                        test_u2b[user_id] = test_basket_ids
        
        # Validate test basket mappings
        unmappable_baskets = []
        train_basket_set = set()
        for baskets in train_u2b.values():
            train_basket_set.update(baskets)
        
        if test_u2b:
            # Next-basket prediction: check test_u2b mappings
            test_basket_set = set()
            for baskets in test_u2b.values():
                test_basket_set.update(baskets)
                
            for basket_id in test_b2i.keys():
                if basket_id not in test_basket_set:
                    unmappable_baskets.append(basket_id)
        else:
            # Within-basket prediction: check if test baskets exist in training
            for basket_id in test_b2i.keys():
                if basket_id not in train_basket_set:
                    unmappable_baskets.append(basket_id)
        
        # Generate statistics
        all_users = set(train_u2b.keys())
        all_train_baskets = train_basket_set
        all_test_baskets = set(test_b2i.keys())
        all_items = set()
        for items in train_b2i.values():
            all_items.update(items)
        for items in test_b2i.values():
            all_items.update(items)
        
        validation_results['statistics'] = {
            'n_users': len(all_users),
            'n_train_baskets': len(all_train_baskets),
            'n_test_baskets': len(all_test_baskets),
            'n_items': len(all_items),
            'n_unmappable_baskets': len(unmappable_baskets),
            'unmappable_rate': len(unmappable_baskets) / len(all_test_baskets) if all_test_baskets else 0.0
        }
        
        # Generate warnings and recommendations
        if unmappable_baskets:
            rate = len(unmappable_baskets) / len(all_test_baskets) * 100
            validation_results['warnings'].append(
                f"{len(unmappable_baskets)}/{len(all_test_baskets)} test baskets ({rate:.1f}%) cannot be mapped to users"
            )
            validation_results['recommendations'].append(
                "Consider filtering unmappable test baskets or investigating data preprocessing steps"
            )
            
            if rate > 10:
                validation_results['warnings'].append(
                    f"High unmappable rate ({rate:.1f}%) may significantly impact evaluation accuracy"
                )
                validation_results['status'] = 'warning'
        
        # Check for overlapping basket IDs between train and test
        overlapping_baskets = all_train_baskets.intersection(all_test_baskets)
        if overlapping_baskets:
            validation_results['warnings'].append(
                f"{len(overlapping_baskets)} basket IDs appear in both training and test sets"
            )
            validation_results['recommendations'].append(
                "Verify this is expected for your prediction task (within-basket vs next-basket)"
            )
        
        # Check for missing test_u2b.txt for next-basket prediction
        if not test_u2b_file.exists():
            validation_results['warnings'].append(
                "test_u2b.txt not found - assuming within-basket prediction task"
            )
            validation_results['recommendations'].append(
                "For next-basket prediction, ensure test_u2b.txt contains user-to-test-basket mappings"
            )
        
        logger.info(f"Data validation completed: {validation_results['status']}")
        
    except Exception as e:
        validation_results['status'] = 'error'
        validation_results['errors'].append(f"Validation failed: {str(e)}")
        logger.error(f"Data validation error: {e}")
    
    return validation_results


def print_validation_report(validation_results: Dict[str, any]) -> None:
    """
    Print a formatted validation report.
    
    Args:
        validation_results: Results from validate_data_files()
    """
    print("\n" + "="*50)
    print("DATA VALIDATION REPORT")
    print("="*50)
    
    # Status
    status = validation_results['status'].upper()
    status_color = {'VALID': '✅', 'WARNING': '⚠️', 'ERROR': '❌'}
    print(f"Status: {status_color.get(status, '❓')} {status}")
    
    # Statistics
    if validation_results['statistics']:
        print(f"\nDataset Statistics:")
        stats = validation_results['statistics']
        print(f"  Users: {stats.get('n_users', 'N/A'):,}")
        print(f"  Train Baskets: {stats.get('n_train_baskets', 'N/A'):,}")
        print(f"  Test Baskets: {stats.get('n_test_baskets', 'N/A'):,}")
        print(f"  Items: {stats.get('n_items', 'N/A'):,}")
        
        if stats.get('n_unmappable_baskets', 0) > 0:
            print(f"  Unmappable Test Baskets: {stats['n_unmappable_baskets']:,} ({stats['unmappable_rate']*100:.1f}%)")
    
    # Errors
    if validation_results['errors']:
        print(f"\n❌ Errors ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            print(f"   • {error}")
    
    # Warnings
    if validation_results['warnings']:
        print(f"\n⚠️  Warnings ({len(validation_results['warnings'])}):")
        for warning in validation_results['warnings']:
            print(f"   • {warning}")
    
    # Recommendations
    if validation_results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"   • {rec}")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        # Default to current directory data
        data_path = Path("data/groceries_data_v0")
    
    if data_path.exists():
        results = validate_data_files(data_path)
        print_validation_report(results)
    else:
        print(f"Error: Data path {data_path} does not exist")
        print("Usage: python3 utils/data_validation.py [data_path]")