import csv
import json
from pathlib import Path
import numpy as np

def write_submission(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target"])
        for row_id, target in rows:
            writer.writerow([int(row_id), int(target)])
    print(f"wrote {filename}: {len(rows)} rows")

def _find(name):
    here = Path(name)
    return here if here.exists() else Path("data") / name

grid = np.load(_find("grid.npy"))
offsets = np.load(_find("offsets.npy"))
left = np.load(_find("left.npy"))
right = np.load(_find("right.npy"))
queries = json.loads(_find("queries.json").read_text(encoding="utf-8"))

windows = queries["windows"]
rows = queries["rows"]
peak_bands = queries["peak_bands"]
bcast = queries["bcast"]
matmul = queries["matmul"]

# Part 1
part1 = [[i, 0] for i in range(len(windows))]
for i in range(len(windows)):
    r0, r1, c0, c1 = windows[i]
    part1[i][1] = int(np.sum(grid[r0:r1, c0:c1]))
write_submission("part1_submission.csv", part1)

# Part 2
part2 = [[i, 0] for i in range(len(rows) + len(peak_bands))]
for i, r in enumerate(rows):
    part2[i][1] = int(np.argmax(grid[r]))

for i, band in enumerate(peak_bands):
    c0, c1 = band
    band_slice = grid[:, c0:c1]
    band_flat_idx = np.argmax(band_slice)
    r_band, c_band = np.unravel_index(band_flat_idx, band_slice.shape)
    c_grid = c_band + c0
    r_grid = r_band
    grid_flat_idx = r_grid * grid.shape[1] + c_grid
    part2[len(rows) + i][1] = int(grid_flat_idx)

write_submission("part2_submission.csv", part2)

# Part 3
part3 = [[i, 0] for i in range(len(bcast))]
grid_sub = grid - offsets
for i, (r, c) in enumerate(bcast):
    val = grid_sub[r, c]
    part3[i][1] = int(np.round(val * 100))
write_submission("part3_submission.csv", part3)

# Part 4
part4 = [[i, 0] for i in range(len(matmul))]
product = left @ right
for k, (i, j) in enumerate(matmul):
    val = product[i, j]
    part4[k][1] = int(np.round(val * 100))
write_submission("part4_submission.csv", part4)
