"""Block: everything a coarse method may look at, for one photo, computed once.

A coarse method is a function  score_map(inputs) -> tensor (rows, cols)  with values in (-1, 1) and -1 where a position is unusable.
`inputs` is the dict made here:
  maps      name -> (rows, cols) score map, all on the same grid
  z         (rows, cols, K) robust z-score of every map, in the order of `names`
  usable    (rows, cols) bool, positions that are inside the search disc and on the map
  context   (2K + 1,) numbers per photo for the gated fusion (peak height and gap per map, ground pixel size)
  names     the K map names
  geometry  grid description (level, x_m, y_m, inside, side), needed by pick_peaks
"""
from train.maps.map_bank import compute_maps, z_stack


def make_inputs(canvas, valid, level, prior_m, families, names, extra_maps=None, extra_names=()):
    """extra_maps(maps, geometry, canvas, valid, level, prior_m) may add maps that compute_maps cannot make; list their names in extra_names."""
    base_names = [name for name in names if name not in extra_names]
    maps, geometry = compute_maps(canvas, valid, level, prior_m, families, base_names)
    if extra_maps is not None:
        extra_maps(maps, geometry, canvas, valid, level, prior_m)
    z, usable, context = z_stack(maps, geometry, names)
    return dict(maps=maps, z=z, usable=usable, context=context, names=list(names), geometry=geometry)
