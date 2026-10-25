"""Offline round-trip smoke check with a generated, non-sensitive animation."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from prepare_grid_data import gif_to_grid, grid_to_gif

with TemporaryDirectory() as temporary:
    path = Path(temporary)
    frames = [Image.new("RGB", (24, 24), (index * 24, 40, 200)) for index in range(8)]
    source = path / "synthetic.gif"
    frames[0].save(source, save_all=True, append_images=frames[1:], duration=100, loop=0)
    grid = gif_to_grid(str(source), frame_size=24)
    assert grid.size == (96, 48), grid.size
    target = path / "roundtrip.gif"
    grid_to_gif(grid, frame_size=24, output_path=str(target))
    with Image.open(target) as result:
        assert result.n_frames == 8, result.n_frames
    print("PASS: synthetic GIF -> 8-frame grid -> GIF; no network, GPU or database")
