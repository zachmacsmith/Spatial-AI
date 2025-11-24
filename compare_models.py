#!/usr/bin/env python3
"""
Compare Models - Standalone Script

Interactive tool to compare different model configurations.
Combines accuracy and performance data to help select the best model.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from post_processing.model_comparison import compare_models, generate_comparison_report
from video_processing.batch_comparison import BatchRegistry
from video_processing.batch_parameters import BatchParameters


def main():
    print("="*60)
    print("MODEL COMPARISON TOOL")
    print("="*60)
    
    # Load available batches
    registry = BatchRegistry()
    all_batches = registry.get_all_batch_ids()
    
    if not all_batches:
        print("\n✗ No batches found in batch tracking directory")
        print("  Run batch_process.py first to generate some batches")
        return
    
    print(f"\nFound {len(all_batches)} tracked batch(es):")
    print()
    
    # Display batches with configuration details
    for i, batch_id in enumerate(all_batches, 1):
        try:
            params = BatchParameters.from_batch_id(batch_id)
            print(f"  {i}. {batch_id}")
            print(f"     Config: {params.config_name}")
            print(f"     LLM: {params.llm_provider.value} ({params.llm_model})")
            print(f"     CV: {params.cv_model.value}")
            print()
        except Exception as e:
            print(f"  {i}. {batch_id}")
            print(f"     (Could not load config: {e})")
            print()
    
    # Select batches to compare
    print("Select batches to compare:")
    selection = input("Enter batch numbers (comma-separated, e.g., 1,3,5): ").strip()
    
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        batch_ids = [all_batches[i] for i in indices]
    except (ValueError, IndexError) as e:
        print(f"\n✗ Invalid selection: {e}")
        return
    
    if len(batch_ids) < 2:
        print("\n✗ Please select at least 2 batches to compare")
        return
    
    print(f"\nWill compare {len(batch_ids)} batches:")
    for batch_id in batch_ids:
        print(f"  - {batch_id}")
    
    # Confirm
    confirm = input("\nContinue? (Y/N): ").strip().upper()
    if confirm != 'Y':
        print("Cancelled")
        return
    
    # Choose comparison type
    print("\nComparison options:")
    print("  1. Accuracy only")
    print("  2. Performance only")
    print("  3. Full comparison (accuracy + performance)")
    
    comp_type = input("Select option (1/2/3, default=3): ").strip()
    
    if comp_type == '1':
        include_accuracy = True
        include_performance = False
    elif comp_type == '2':
        include_accuracy = False
        include_performance = True
    else:
        include_accuracy = True
        include_performance = True
    
    # Run comparison
    print("\n" + "="*60)
    print("Running comparison...")
    print("="*60)
    
    try:
        # Generate full report
        report_path = generate_comparison_report(batch_ids)
        
        print("\n" + "="*60)
        print("✓ COMPARISON COMPLETE!")
        print("="*60)
        print(f"\nReport saved to: {report_path}")
        print("\nOpen the HTML report in your browser to view:")
        print(f"  file://{os.path.abspath(report_path)}")
        
        # Ask if user wants to open in browser
        open_browser = input("\nOpen report in browser now? (Y/N): ").strip().upper()
        if open_browser == 'Y':
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(report_path))
            print("✓ Opened in browser")
        
    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
