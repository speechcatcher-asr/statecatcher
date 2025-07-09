def parse_timestamp(timestamp):
    # Split the timestamp into parts
    parts = timestamp.split(":")

    # Handle milliseconds
    last_part = parts[-1].split(".")
    seconds = int(last_part[0])
    milliseconds = float("0." + last_part[1]) if len(last_part) > 1 else 0

    # Calculate total time in seconds
    if len(parts) == 3:  # Format: HH:MM:SS.sss
        h = int(parts[0])
        m = int(parts[1])
        total_seconds = h * 3600 + m * 60 + seconds + milliseconds
    elif len(parts) == 2:  # Format: MM:SS.sss
        m = int(parts[0])
        total_seconds = m * 60 + seconds + milliseconds
    else:
        raise ValueError(f"Timestamp format is incorrect: {timestamp}")

    return total_seconds

def vtt_to_segments_with_text(vtt_text):
    segments = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            try:
                # Parse the time range
                parts = line.split("-->")
                start_time = parts[0].strip()
                end_time = parts[1].strip()

                start_sec = parse_timestamp(start_time)
                end_sec = parse_timestamp(end_time)

                # Extract the corresponding text
                i += 1
                text = []
                while i < len(lines) and lines[i].strip() != "":
                    text.append(lines[i].strip())
                    i += 1
                text = " ".join(text)
                segments.append((start_sec, end_sec, text))
            except Exception as e:
                print(f"Error parsing segment: {e}")
                i += 1
                continue
        else:
            i += 1
    return segments

if __name__ == "__main__":
    # Example VTT content
    vtt_content = """WEBVTT
00:00.000 --> 00:00:29.980
Thank you for listening.
"""
    # Simulate parsing
    segments = vtt_to_segments_with_text(vtt_content)
    print(segments)

