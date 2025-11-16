import os
from TestingClass import process_video  # CHANGED: Import from TestingClass instead

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

# Process each video
for i, video_name in enumerate(videos_to_process, 1):
    print(f"\n{'='*50}")
    print(f"Processing {i}/{len(videos_to_process)}: {video_name}")
    print(f"{'='*50}\n")
    
    try:
        csv_path = process_video(video_name)
        print(f"\n✓ Successfully processed {video_name}")
        print(f"   Output: {csv_path}")
    except Exception as e:
        print(f"\n✗ ERROR processing {video_name}: {e}")
        continue_anyway = input("Continue with remaining videos? (Y/N): ").strip().upper()
        if continue_anyway != 'Y':
            break

print("\n" + "="*50)
print("Batch processing complete!")
print("="*50)