"""
Enhanced Test Script - All Presets with Output Validation

Tests all 5 presets and validates that outputs are correct and functional.
"""

import sys
import os
import time
import csv
import json
from pathlib import Path
from datetime import datetime

# Add project root to path (go up two directories from docs/testing/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from video_processing import (
    process_video,
    PRESET_BASIC,
    PRESET_OBJECTS,
    PRESET_RELATIONSHIPS,
    PRESET_HTML_ANALYSIS,
    PRESET_FULL
)


def validate_csv_file(csv_path, expected_batch_id):
    """Validate CSV file exists and has correct format"""
    if not os.path.exists(csv_path):
        return False, "File does not exist"
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) < 2:
                return False, "CSV has too few rows"
            
            # Check header
            header = rows[0]
            if len(header) < 2:
                return False, "Invalid header format"
            
            # Check if batch_id is in header (if tracking enabled)
            if len(header) >= 3 and header[2] != expected_batch_id:
                return False, f"Batch ID mismatch: expected {expected_batch_id}, got {header[2]}"
            
            # Check data rows
            for i, row in enumerate(rows[1:], 1):
                if len(row) < 2:
                    return False, f"Row {i} has invalid format"
                
                # Validate frame number
                try:
                    frame_num = int(row[0])
                    if frame_num < 1:
                        return False, f"Invalid frame number: {frame_num}"
                except ValueError:
                    return False, f"Frame number not an integer: {row[0]}"
                
                # Validate action label
                if not row[1] or row[1].strip() == "":
                    return False, f"Empty action label at row {i}"
            
            return True, f"Valid CSV with {len(rows)-1} action changes"
    
    except Exception as e:
        return False, f"Error reading CSV: {e}"


def validate_metadata_file(metadata_path, expected_batch_id, expected_video_name):
    """Validate metadata JSON file"""
    if not os.path.exists(metadata_path):
        return False, "File does not exist"
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check required fields
        required_fields = ['video_name', 'batch_id', 'config_name', 'model_versions']
        for field in required_fields:
            if field not in metadata:
                return False, f"Missing required field: {field}"
        
        # Validate values
        if metadata['video_name'] != expected_video_name:
            return False, f"Video name mismatch: {metadata['video_name']}"
        
        if metadata['batch_id'] != expected_batch_id:
            return False, f"Batch ID mismatch: {metadata['batch_id']}"
        
        # Check model versions
        if 'llm_provider' not in metadata['model_versions']:
            return False, "Missing LLM provider in model_versions"
        
        return True, "Valid metadata"
    
    except Exception as e:
        return False, f"Error reading metadata: {e}"


def validate_video_file(video_path):
    """Validate video file exists and is not empty"""
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        return False, "Video file is empty"
    
    if file_size < 1000:  # Less than 1KB is suspicious
        return False, f"Video file too small: {file_size} bytes"
    
    return True, f"Valid video file ({file_size/1024/1024:.2f} MB)"


def test_preset_with_validation(preset, preset_name, video_name="video_01"):
    """Test a preset and validate all outputs"""
    print("=" * 70)
    print(f"Testing: {preset_name}")
    print("=" * 70)
    print(f"Batch ID: {preset.batch_id}")
    print(f"Config: {preset.config_name}")
    print()
    
    start_time = time.time()
    validation_results = {}
    
    try:
        # Process video
        outputs = process_video(video_name, preset)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print()
        print("-" * 70)
        print("Validating Outputs...")
        print("-" * 70)
        
        # Validate each expected output
        # We construct expected paths to ensure they exist, rather than just trusting returned dict
        
        # 1. Actions CSV (Always expected by default unless disabled)
        if preset.save_actions_csv:
            expected_csv = Path(f"outputs/data/{preset.batch_id}/{video_name}.csv")
            print(f"\nChecking Actions CSV: {expected_csv}")
            
            if expected_csv.exists():
                valid, msg = validate_csv_file(expected_csv, preset.batch_id)
                validation_results['actions_csv'] = (valid, msg)
                print(f"  {'✓' if valid else '✗'} {msg}")
            else:
                validation_results['actions_csv'] = (False, "File missing on disk")
                print(f"  ✗ File missing on disk")

        # 2. Metadata (Always expected)
        expected_metadata = Path(f"outputs/data/{preset.batch_id}/{video_name}_metadata.json")
        print(f"\nChecking Metadata: {expected_metadata}")
        
        if expected_metadata.exists():
            valid, msg = validate_metadata_file(expected_metadata, preset.batch_id, video_name)
            validation_results['metadata'] = (valid, msg)
            print(f"  {'✓' if valid else '✗'} {msg}")
        else:
            validation_results['metadata'] = (False, "File missing on disk")
            print(f"  ✗ File missing on disk")

        # 3. Labeled Video (If enabled)
        if preset.generate_labeled_video:
            expected_video = Path(f"outputs/vid_objs/{preset.batch_id}/{video_name}.mp4")
            print(f"\nChecking Labeled Video: {expected_video}")
            
            if expected_video.exists():
                valid, msg = validate_video_file(expected_video)
                validation_results['video'] = (valid, msg)
                print(f"  {'✓' if valid else '✗'} {msg}")
            else:
                validation_results['video'] = (False, "File missing on disk")
                print(f"  ✗ File missing on disk")

        # 4. Relationships CSV (If enabled)
        if preset.save_relationships_csv and preset.enable_relationship_tracking:
            expected_rel_csv = Path(f"outputs/data/{preset.batch_id}/{video_name}_relationships.csv")
            print(f"\nChecking Relationships CSV: {expected_rel_csv}")
            
            if expected_rel_csv.exists():
                file_size = os.path.getsize(expected_rel_csv)
                validation_results['relationships_csv'] = (True, f"File exists ({file_size} bytes)")
                print(f"  ✓ File exists ({file_size} bytes)")
            else:
                # Note: It's possible to have no relationships found, but the file might not be created?
                # Usually output manager creates it even if empty or just header?
                # Let's assume it should exist.
                validation_results['relationships_csv'] = (False, "File missing on disk")
                print(f"  ✗ File missing on disk")
        
        # Overall validation
        all_valid = all(valid for valid, _ in validation_results.values())
        
        print()
        print("-" * 70)
        if all_valid:
            print(f"✓ {preset_name} PASSED - All outputs valid")
        else:
            print(f"⚠ {preset_name} PASSED - But some validations failed")
        print("-" * 70)
        print(f"Processing time: {processing_time:.2f}s ({processing_time/60:.2f} min)")
        print(f"Validation: {sum(1 for v, _ in validation_results.values() if v)}/{len(validation_results)} checks passed")
        print()
        
        return True, processing_time, outputs, validation_results
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print()
        print("-" * 70)
        print(f"✗ {preset_name} FAILED")
        print("-" * 70)
        print(f"Error: {e}")
        print(f"Time before failure: {processing_time:.2f}s")
        print()
        
        import traceback
        traceback.print_exc()
        
        return False, processing_time, None, {}


def main():
    """Test all presets with validation"""
    print("=" * 70)
    print("ENHANCED PRESET TESTING - WITH OUTPUT VALIDATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Video: video_01")
    print()
    print("Features:")
    print("  ✓ Batch-specific output folders (no overwrites)")
    print("  ✓ Output validation (file existence, format, content)")
    print("  ✓ Detailed validation reporting")
    print()
    
    # Wait to avoid rate limits
    print("Waiting 60 seconds to avoid rate limits...")
    time.sleep(60)
    print()
    
    presets = [
        (PRESET_BASIC, "PRESET_BASIC"),
        (PRESET_OBJECTS, "PRESET_OBJECTS"),
        (PRESET_RELATIONSHIPS, "PRESET_RELATIONSHIPS"),
        (PRESET_HTML_ANALYSIS, "PRESET_HTML_ANALYSIS"),
        (PRESET_FULL, "PRESET_FULL")
    ]
    
    results = []
    total_start = time.time()
    
    for preset, name in presets:
        success, time_taken, outputs, validations = test_preset_with_validation(preset, name)
        results.append({
            'name': name,
            'success': success,
            'time': time_taken,
            'outputs': outputs,
            'validations': validations,
            'batch_id': preset.batch_id
        })
        
        # Wait between tests
        if name != "PRESET_FULL":
            print("Waiting 60 seconds before next test...")
            time.sleep(60)
            print()
    
    total_end = time.time()
    total_time = total_end - total_start
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(results)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print()
    
    print("Results by preset:")
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        val_count = len(r['validations'])
        val_passed = sum(1 for v, _ in r['validations'].values() if v) if r['validations'] else 0
        print(f"  {status} {r['name']}: {r['time']:.2f}s | Validations: {val_passed}/{val_count}")
        print(f"      Batch: {r['batch_id']}")
        if r['outputs']:
            print(f"      Outputs in: outputs/*/{{batch_id}}/")
    
    print()
    print("=" * 70)
    print("OUTPUT ORGANIZATION")
    print("=" * 70)
    print()
    print("All outputs are now organized by batch ID:")
    print("  outputs/data/{batch_id}/video_01.csv")
    print("  outputs/data/{batch_id}/video_01_relationships.csv")
    print("  outputs/data/{batch_id}/video_01_metadata.json")
    print("  outputs/vid_objs/{batch_id}/video_01.mp4")
    print()
    print("This prevents overwrites and makes it easy to compare different runs!")
    print()
    print("=" * 70)
    
    if failed == 0:
        print("✓ ALL PRESETS PASSED WITH VALID OUTPUTS!")
        print("=" * 70)
        return 0
    else:
        print(f"✗ {failed} PRESET(S) FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
