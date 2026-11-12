"""Drone-view simulator: a satellite map goes in, a simulated thermal drone photo (camera perspective) comes out.

Blocks, one job each (swap any of them):
  camera_model  camera numbers (focal, centre, lens distortion), scaled copies for small test images
  pose          where the drone is and how the camera points; made from a photo item or from numbers
  ground_rays   for every camera pixel, the point on the ground it sees (flat ground or a height map)
  map_sampler   reads the needed map piece and samples it into the camera image with area averaging
  appearance    what the ground looks like in thermal (grey map -> thermal-like grey values); swappable
  sensor        blur, noise, stripes and 8-bit quantisation of the camera
  render        puts the blocks together: render_view(...) -> image, valid mask, metadata
  dataset       random poses and saving examples (png + json) with their metadata
"""
