#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from auxiliary.vispy_manager import VispyManager


class LaserScanComp:
  """Class that creates and handles a side-by-side pointcloud comparison"""

  def __init__(self, scans, scan_names, label_names, offset=0, images=True, link=False):
    self.scan_a_view = None
    self.scan_a_vis = None
    self.scan_b_view = None
    self.scan_b_vis = None
    self.scan_a, self.scan_b = scans
    self.scan_a_names, self.scan_b_names = scan_names
    self.label_a_names, self.label_b_names = label_names
    self.offset = offset
    self.total = len(self.scan_a_names)
    self.images = images
    self.link = link
    self.vispy_manager = VispyManager(self.key_press, self.draw)
    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # new canvas prepared for visualizing data
    self.scan_a_view, self.scan_a_vis = self.vispy_manager.add_viewbox(0, 0)
    self.scan_b_view, self.scan_b_vis = self.vispy_manager.add_viewbox(0, 1)

    if self.link:
      self.scan_a_view.camera.link(self.scan_b_view.camera)

  def update_scan(self):
    self.scan_a.open_scan(self.scan_a_names[self.offset])
    self.scan_a.open_label(self.label_a_names[self.offset])
    self.scan_a.colorize()
    self.scan_a_vis.set_data(self.scan_a.points,
                          face_color=self.scan_a.sem_label_color[..., ::-1],
                          edge_color=self.scan_a.sem_label_color[..., ::-1],
                          size=1)

    self.scan_b.open_scan(self.scan_b_names[self.offset])
    self.scan_b.open_label(self.label_b_names[self.offset])
    self.scan_b.colorize()
    self.scan_b_vis.set_data(self.scan_b.points,
                          face_color=self.scan_b.sem_label_color[..., ::-1],
                          edge_color=self.scan_b.sem_label_color[..., ::-1],
                          size=1)

  def key_press(self, event):
    self.vispy_manager.block_key_press()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.vispy_manager.destroy()
    pass

  def draw(self, event):
    if self.vispy_manager.key_press_blocked():
      self.vispy_manager.unblock_key_press()

  def run(self):
    self.vispy_manager.run()
