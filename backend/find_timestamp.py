from PIL import Image, ImageSequence
import numpy as np

# Load the GIF
gif = Image.open('delhi_radar.gif')
frames = list(ImageSequence.Iterator(gif))
last_frame = frames[-1].convert('RGB')
arr = np.array(last_frame)

print(f"Image size: {last_frame.size}")
print()

# STEP 1: Find where top brown panel ends
# Brown panel has RGB roughly (150-200, 100-150, 50-100)
# Green radar starts after the panel
# Scan from top to bottom on center column

print("Scanning for top panel boundary...")
center_x = arr.shape[1] // 2
TOP_BOUNDARY = None
for y in range(50, 300):
    r, g, b = arr[y, center_x]
    # Green terrain color
    is_green = (60 < r < 180) and (100 < g < 200) and (30 < b < 120)
    if is_green:
        print(f"Top panel ends at Y = {y}")
        TOP_BOUNDARY = y
        break

# STEP 2: Find where right panel starts
# Right panel is gray/dark background
# Scan from right to left on middle row

print("Scanning for right panel boundary...")
mid_y = arr.shape[0] // 2
RIGHT_BOUNDARY = None
for x in range(arr.shape[1]-1, 400, -1):
    r, g, b = arr[mid_y, x]
    is_green = (60 < r < 180) and (100 < g < 200) and (30 < b < 120)
    if is_green:
        print(f"Right panel starts at X = {x}")
        RIGHT_BOUNDARY = x
        break

# STEP 3: Find timestamp location
# Timestamp = WHITE or BRIGHT text
# on DARK background
# Located in right panel area
# Scan right panel for bright white pixels

print()
print("Scanning for bright white text in right panel...")

if RIGHT_BOUNDARY is None:
    raise RuntimeError("Could not detect RIGHT_BOUNDARY automatically.")

# Right panel is from RIGHT_BOUNDARY to end
# Scan this region for white pixels
white_pixels = []
for y in range(100, 400):
    for x in range(RIGHT_BOUNDARY, arr.shape[1]):
        r, g, b = arr[y, x]
        # White text = all channels high
        if r > 200 and g > 200 and b > 200:
            white_pixels.append((x, y))

if white_pixels:
    xs = [p[0] for p in white_pixels]
    ys = [p[1] for p in white_pixels]
    print(f"White pixels found in right panel:")
    print(f"  X range: {min(xs)} to {max(xs)}")
    print(f"  Y range: {min(ys)} to {max(ys)}")

    # Find densest cluster (that's the timestamp)
    # Group by Y rows
    y_counts = {}
    for x, y in white_pixels:
        y_counts[y] = y_counts.get(y, 0) + 1

    # Find rows with most white pixels
    sorted_rows = sorted(y_counts.items(),
                        key=lambda x: x[1],
                        reverse=True)

    print()
    print("Top 10 rows with most white pixels:")
    for row_y, count in sorted_rows[:10]:
        print(f"  Y={row_y}: {count} white pixels")

    # The timestamp rows will have many white pixels
    # Find the Y range of timestamp
    top_rows = [y for y, c in sorted_rows[:20]]
    ts_y_min = min(top_rows)
    ts_y_max = max(top_rows)
    ts_x_min = min(xs)
    ts_x_max = max(xs)

    print()
    print(f"Estimated timestamp region:")
    print(f"  X: {ts_x_min} to {ts_x_max}")
    print(f"  Y: {ts_y_min} to {ts_y_max}")

    # Save the timestamp crop
    ts_crop = last_frame.crop((
        ts_x_min - 5,
        ts_y_min - 5,
        ts_x_max + 5,
        ts_y_max + 5
    ))
    ts_crop.save('debug_timestamp_found.png')
    print()
    print("Saved: debug_timestamp_found.png")
    print("Check this image to verify timestamp!")

# STEP 4: Also save right panel crop for inspection
right_panel = last_frame.crop((
    RIGHT_BOUNDARY, 100,
    arr.shape[1], 400
))
right_panel.save('debug_right_panel.png')
print()
print("Saved: debug_right_panel.png")

