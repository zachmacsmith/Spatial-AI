import os
import time
import csv
from datetime import datetime
from TestingClass import process_video
from KeyFrameClassifier import process_video as generate_keyframes

# Default videos to process (used if user doesn't examine folder)
videos_to_process = [
    "video_01",
    "video_02",
    "video_05"
]

# Ask which processing mode to use
use_objects = input("Use object identification? (Y/N): ").strip().upper()

if use_objects == 'Y':
    from TestingClass2 import process_video
    processing_mode = "with object identification"
else:
    from TestingClass import process_video
    processing_mode = "without object identification"

print(f"\nRunning {processing_mode}")

# Ask if user wants to examine videos folder
examine = input("Examine videos folder to select files? (Y/N): ").strip().upper()

if examine == 'Y' or examine == 'A':
    videos_folder = "videos"
    
    # Find all video files
    video_files = [f.replace('.mp4', '') for f in os.listdir(videos_folder) if f.endswith('.mp4')]
    video_files.sort()
    
    print(f"\nFound {len(video_files)} videos in {videos_folder}/")
    print()
    if examine == 'A':
        # Process all videos with keyframes
        for video_name in video_files:
            keyframes_folder = f"keyframes/keyframes{video_name}"
            has_keyframes = os.path.exists(keyframes_folder) and len(os.listdir(keyframes_folder)) > 0
            
            if has_keyframes:
                videos_to_process.append(video_name)
                print(f"✓ {video_name} - added to queue")
            else:
                videos_to_process.append(video_name)
                print(f"✓ {video_name} - added to queue (will generate keyframes)")
    else:
        # Ask Y/N for each video
        videos_to_process = []
        for video_name in video_files:
            # Check if keyframes exist
            keyframes_folder = f"keyframes/keyframes{video_name}"
            has_keyframes = os.path.exists(keyframes_folder) and len(os.listdir(keyframes_folder)) > 0
            
            if not has_keyframes:
                print(f"⚠️  {video_name} - no keyframes found (will be generated)")
                
            response = input(f"Include {video_name}? (Y/N): ").strip().upper()
            if response == 'Y':
                videos_to_process.append(video_name)
    
    if not videos_to_process:
        print("No videos selected")
        exit()

# Show what will be processed
print(f"\nWill process {len(videos_to_process)} videos: {videos_to_process}")
confirm = input("Continue? (Y/N): ").strip().upper()

if confirm != 'Y':
    print("Cancelled")
    exit()

# Get batch note
batch_note = input("Enter batch note (optional, press Enter to skip): ").strip()

# Ask if user wants to benchmark after processing
run_benchmark_flag = input("Run benchmark after processing? (Y/N): ").strip().upper()

# Determine batch number and handle CSV migration
timing_file = 'Outputs/timing_results.csv'
os.makedirs('Outputs', exist_ok=True)

if os.path.exists(timing_file):
    # Check if CSV needs migration (missing batch_note column)
    with open(timing_file, 'r') as f:
        first_line = f.readline().strip()
        needs_migration = 'batch_note' not in first_line
    
    if needs_migration:
        print("\nMigrating CSV to include batch_note column...")
        # Read all existing rows
        with open(timing_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Add batch_note to header and empty values to all data rows
        rows[0].append('batch_note')
        for i in range(1, len(rows)):
            rows[i].append('')
        
        # Write back
        with open(timing_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print("Migration complete.")
    
    # Read to find max batch number
    with open(timing_file, 'r') as f:
        reader = csv.DictReader(f)
        batches = [int(row['batch']) for row in reader]
        batch_number = max(batches) + 1 if batches else 1
else:
    batch_number = 1

print(f"\nBatch number: {batch_number} ({processing_mode})")
if batch_note:
    print(f"Batch note: {batch_note}")

# Open timing file for appending
file_exists = os.path.exists(timing_file)
timing_file_handle = open(timing_file, 'a', newline='')
timing_writer = csv.writer(timing_file_handle)

if not file_exists:
    timing_writer.writerow(['batch', 'timestamp', 'video_name', 'video_duration_sec', 'processing_time_sec', 'processing_time_min', 'speed_ratio', 'batch_note'])

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
timing_results = {}

# Process each video
for i, video_name in enumerate(videos_to_process, 1):
    print(f"\n{'='*50}")
    print(f"Processing {i}/{len(videos_to_process)}: {video_name}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    try:
        # Check/Generate keyframes
        keyframes_folder = f"keyframes/keyframes{video_name}"
        if not os.path.exists(keyframes_folder) or len(os.listdir(keyframes_folder)) == 0:
             print(f"Generating keyframes for {video_name}...")
             generate_keyframes(video_name)

        csv_path = process_video(video_name)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Extract video duration from generated CSV
        with open(csv_path, 'r') as data_file:
            first_line = data_file.readline().strip().split(',')
            video_duration = float(first_line[1])
        
        speed_ratio = processing_time / video_duration
        timing_results[video_name] = {
            'processing_time': processing_time,
            'video_duration': video_duration,
            'speed_ratio': speed_ratio
        }
        
        # Write to timing CSV immediately
        timing_writer.writerow([
            batch_number,
            timestamp,
            video_name,
            video_duration,
            processing_time,
            processing_time/60,
            speed_ratio,
            batch_note
        ])
        timing_file_handle.flush()  # Ensure it's written immediately
        
        print(f"\n✓ Successfully processed {video_name}")
        print(f"   Output: {csv_path}")
        print(f"   Video duration: {video_duration:.2f}s")
        print(f"   Processing time: {processing_time:.2f}s ({processing_time/60:.2f} min)")
        print(f"   Speed ratio: {speed_ratio:.2f}x realtime")
        
    except Exception as e:
        print(f"\n✗ ERROR processing {video_name}: {e}")
        continue_anyway = input("Continue with remaining videos? (Y/N): ").strip().upper()
        if continue_anyway != 'Y':
            break

# Close timing file
timing_file_handle.close()

# Print batch summary
if timing_results:
    print("\n" + "="*50)
    print(f"BATCH {batch_number} SUMMARY")
    print("="*50)
    for video_name, metrics in timing_results.items():
        print(f"{video_name}: {metrics['processing_time']:.2f}s ({metrics['speed_ratio']:.2f}x realtime)")
    
    total_processing = sum(m['processing_time'] for m in timing_results.values())
    total_video = sum(m['video_duration'] for m in timing_results.values())
    avg_processing = total_processing / len(timing_results)
    avg_speed_ratio = total_processing / total_video
    
    print(f"\nTotal processing time: {total_processing:.2f}s ({total_processing/60:.2f} min)")
    print(f"Total video duration: {total_video:.2f}s ({total_video/60:.2f} min)")
    print(f"Average processing time: {avg_processing:.2f}s")
    print(f"Average speed ratio: {avg_speed_ratio:.2f}x realtime")
    print(f"\nTiming data saved to: {timing_file}")

print("\n" + "="*50)
print("Batch processing complete!")
print("="*50)

# Run benchmark if requested
if run_benchmark_flag == 'Y':
    print("\n" + "="*50)
    print("Starting Benchmark")
    print("="*50)
    
    from Benchmark import run_benchmark
    
    # Determine which model data directory to use
    if use_objects == 'Y':
        model_data_dir = "Outputs/Data2/"
    else:
        model_data_dir = "Outputs/Data/"
    
    # Get benchmark metadata
    benchmark_batch_name = input("\nBenchmark batch name: ").strip()
    if not benchmark_batch_name:
        benchmark_batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_version = input("Model version (e.g., TestingClass2_v1): ").strip()
    if not model_version:
        if use_objects == 'Y':
            model_version = "TestingClass2"
        else:
            model_version = "TestingClass"
    
    benchmark_notes = input("Benchmark notes (optional): ").strip()
    
    # Run benchmark
    try:
        results = run_benchmark(
            videos_to_process=videos_to_process,
            batch_name=benchmark_batch_name,
            model_version=model_version,
            notes=benchmark_notes,
            model_data_dir=model_data_dir
        )
        
        if results:
            print("\n" + "="*50)
            print("BENCHMARK COMPLETE")
            print("="*50)
            print(f"Average state accuracy: {results['avg_state_accuracy']:.2%}")
            if results['avg_object_accuracy'] is not None:
                print(f"Average object accuracy: {results['avg_object_accuracy']:.2%}")
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print("All processing complete!")
print("="*50)