import numpy as np

from vein_hog.extract import _one
from vein_hog.hog import HOG_DIM, hog_descriptor
from vein_hog.mask import erode_interior, leaf_mask


def test_hog_dimension_is_dalal_triggs_on_256() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, 1] = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
    feat = hog_descriptor(image)
    assert feat.shape == (HOG_DIM,)
    assert feat.dtype == np.float32
    assert HOG_DIM == 34596

    small = np.zeros((48, 64, 3), dtype=np.uint8)
    small[:, :, 1] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    resized = hog_descriptor(small)
    assert resized.shape == (HOG_DIM,)
    assert resized.dtype == np.float32
    assert np.isfinite(resized).all()


def test_interior_line_is_not_the_disk_boundary() -> None:
    """A pale line inside a green disk must survive; the disk rim must not."""
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    blank_sample = _one(blank)
    assert blank_sample["raw_completeness"] == 0.0

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:256, :256]
    disk = (yy - 128) ** 2 + (xx - 128) ** 2 <= 100 ** 2
    image[disk] = (30, 120, 30)
    image[124:133, 60:196] = (220, 240, 220)
    sample = _one(image, radius=8)
    mask = np.asarray(sample["mask256"])
    interior = np.asarray(sample["interior"])
    skeleton = np.asarray(sample["skeleton"])
    assert int((mask > 0).sum()) > 0
    ring = (mask > 0) & ~(interior > 0)
    assert int((skeleton & ring).sum()) == 0
    line = np.zeros(skeleton.shape, dtype=bool)
    line[124:133, 60:196] = True
    on_line = int((skeleton & line).sum())
    off_line = int((skeleton & ~line).sum())
    assert on_line > 0
    assert on_line > off_line
    assert float(sample["raw_completeness"]) > 0.0
    # The erosion that excludes the rim is wider than a 1-pixel contour.
    assert int((erode_interior(mask, 8) > 0).sum()) < int((mask > 0).sum())

    empty = leaf_mask(np.zeros((0, 4, 3), dtype=np.uint8))
    assert empty.shape == (0, 4)
    assert empty.dtype == np.uint8
    failed = leaf_mask(np.zeros((8, 8), dtype=np.uint8))
    assert failed.shape == (8, 8)
    assert int(failed.max()) == 0
