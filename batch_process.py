import os
import time
import csv
from datetime import datetime
from TestingClass import process_video

# Default videos to process (used if user doesn't examine folder)
videos_to_process = [
    "video_01",
    "video_02",
    "video_05"
]

# Ask if user wants to examine videos folder
examine = input("Examine videos folder to select files? (Y/N): ").strip().upper()

if examine == 'Y':
    videos_folder = "videos"
    
    # Find all video files
    video_files = [f.replace('.mp4', '') for f in os.listdir(videos_folder) if f.endswith('.mp4')]
    video_files.sort()
    
    print(f"\nFound {len(video_files)} videos in {videos_folder}/")
    print()
    
    # Ask Y/N for each video
    videos_to_process = []
    for video_name in video_files:
        # Check if keyframes exist
        keyframes_folder = f"keyframes/keyframes{video_name}"
        has_keyframes = os.path.exists(keyframes_folder) and len(os.listdir(keyframes_folder)) > 0
        
        if not has_keyframes:
            print(f"⚠️  {video_name} - no keyframes found, skipping")
            continue
            
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

# Determine batch number
timing_file = 'Outputs/timing_results.csv'
os.makedirs('Outputs', exist_ok=True)

if os.path.exists(timing_file):
    # Read existing file to find max batch number
    with open(timing_file, 'r') as f:
        reader = csv.DictReader(f)
        batches = [int(row['batch']) for row in reader]
        batch_number = max(batches) + 1 if batches else 1
else:
    batch_number = 1

print(f"\nBatch number: {batch_number}")

# Open timing file for appending
file_exists = os.path.exists(timing_file)
timing_file_handle = open(timing_file, 'a', newline='')
timing_writer = csv.writer(timing_file_handle)

if not file_exists:
    timing_writer.writerow(['batch', 'timestamp', 'video_name', 'video_duration_sec', 'processing_time_sec', 'processing_time_min', 'speed_ratio'])

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
timing_results = {}

# Process each video
for i, video_name in enumerate(videos_to_process, 1):
    print(f"\n{'='*50}")
    print(f"Processing {i}/{len(videos_to_process)}: {video_name}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    try:
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
            speed_ratio
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