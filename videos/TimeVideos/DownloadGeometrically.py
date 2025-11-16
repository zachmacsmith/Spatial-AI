import subprocess
import math
import os

def generate_progressive_clips(url, start_time, end_time, sampling_rate, min_duration=1):
    """
    Downloads full video, then cuts clips locally with ffmpeg.
    """
    
    def parse_time(t):
        if isinstance(t, str):
            parts = list(map(int, t.split(':')))
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            else:
                return parts[0]
        return t
    
    start_sec = parse_time(start_time)
    end_sec = parse_time(end_time)
    total_duration = end_sec - start_sec
    
    if total_duration <= 0:
        print("Error: end_time must be after start_time")
        return
    
    if sampling_rate < 2:
        print("Error: sampling_rate must be at least 2")
        return
    
    # Step 1: Download full video
    full_video = "temp_full_video.mp4"
    if not os.path.exists(full_video):
        print("Downloading full video...")
        subprocess.run([
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", full_video,
            url
        ])
    else:
        print("Using existing full video")
    
    # Step 2: Cut clips with ffmpeg
    growth_factor = (total_duration / min_duration) ** (1 / (sampling_rate - 1))
    
    print(f"\nTotal duration: {total_duration}s")
    print(f"Growth factor: {growth_factor:.4f}")
    print(f"Generating {sampling_rate} clips\n")
    
    for i in range(sampling_rate):
        duration = min_duration * (growth_factor ** i)
        output_file = f"clip_{i+1:03d}_duration_{int(duration)}s.mp4"
        
        print(f"Clip {i+1}/{sampling_rate}: duration {duration:.2f}s")
        
        # ffmpeg: -ss for seek, -t for duration, -c copy for no re-encoding
        subprocess.run([
            "ffmpeg",
            "-ss", str(start_sec),
            "-i", full_video,
            "-t", str(duration),
            "-c", "copy",
            "-y",  # overwrite
            output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("\nDone!")

if __name__ == "__main__":
    VIDEO_URL = "https://www.youtube.com/watch?v=UWp2_DBqqz0"
    START_TIME = "04:00"
    END_TIME = "20:00"
    SAMPLING_RATE = 10
    MIN_DURATION = 1
    
    generate_progressive_clips(VIDEO_URL, START_TIME, END_TIME, SAMPLING_RATE, MIN_DURATION)