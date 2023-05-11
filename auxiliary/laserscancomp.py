#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from auxiliary.vispy_manager import VispyManager
import numpy as np

class LaserScanComp(VispyManager):
  """Class that creates and handles a side-by-side pointcloud comparison"""

  def __init__(self, scans, scan_names, label_names, offset=0, images=True, instances=False, link=False):
    super().__init__(offset, len(scan_names), images, instances)
    self.scan_a_view = None
    self.scan_a_vis = None
    self.scan_b_view = None
    self.scan_b_vis = None
    self.inst_a_view = None
    self.inst_a_vis = None
    self.inst_b_view = None
    self.inst_b_vis = None
    self.img_a_view = None
    self.img_a_vis = None
    self.img_b_view = None
    self.img_b_vis = None
    self.img_inst_a_view = None
    self.img_inst_a_vis = None
    self.img_inst_b_view = None
    self.img_inst_b_vis = None
    self.scan_a, self.scan_b = scans
    self.scan_names = scan_names
    self.label_a_names, self.label_b_names = label_names
    self.link = link
    self.reset()
    self.update_scan()

  def reset(self):
    """prepares the canvas(es) for the visualizer"""
    self.scan_a_view, self.scan_a_vis = super().add_viewbox(0, 0)
    self.scan_b_view, self.scan_b_vis = super().add_viewbox(0, 1)

    if self.link:
      self.scan_a_view.camera.link(self.scan_b_view.camera)

    if self.images:
      self.img_a_view, self.img_a_vis = super().add_image_viewbox(0, 0)
      self.img_b_view, self.img_b_vis = super().add_image_viewbox(1, 0)

      if self.instances:
        self.img_inst_a_view, self.img_inst_a_vis = super().add_image_viewbox(2, 0)
        self.img_inst_b_view, self.img_inst_b_vis = super().add_image_viewbox(3, 0)

    if self.instances:
      self.inst_a_view, self.inst_a_vis = super().add_viewbox(1, 0)
      self.inst_b_view, self.inst_b_vis = super().add_viewbox(1, 1)

      if self.link:
        self.scan_a_view.camera.link(self.inst_a_view.camera)
        self.inst_a_view.camera.link(self.inst_b_view.camera)

  def update_scan(self):
    """updates the scans, images and instances"""
    self.scan_a.open_scan(self.scan_names[self.offset])
    self.scan_a.open_label(self.label_a_names[self.offset])
    self.scan_a.colorize()
    self.scan_a_vis.set_data(self.scan_a.points,
                          face_color=self.scan_a.sem_label_color[..., ::-1],
                          edge_color=self.scan_a.sem_label_color[..., ::-1],
                          size=1)

    self.scan_b.open_scan(self.scan_names[self.offset])
    self.scan_b.open_label(self.label_b_names[self.offset])
    self.scan_b.colorize()
    self.scan_b_vis.set_data(self.scan_b.points,
                          face_color=self.scan_b.sem_label_color[..., ::-1],
                          edge_color=self.scan_b.sem_label_color[..., ::-1],
                          size=1)

    if self.instances:
      self.inst_a_vis.set_data(self.scan_a.points,
                               face_color=self.scan_a.inst_label_color[..., ::-1],
                               edge_color=self.scan_a.inst_label_color[..., ::-1],
                               size=1)
      self.inst_b_vis.set_data(self.scan_b.points,
                               face_color=self.scan_b.inst_label_color[..., ::-1],
                               edge_color=self.scan_b.inst_label_color[..., ::-1],
                               size=1)

    if self.images:
      self.img_a_vis.set_data(self.scan_a.proj_sem_color[..., ::-1])
      self.img_a_vis.update()
      self.img_b_vis.set_data(self.scan_b.proj_sem_color[..., ::-1])
      self.img_b_vis.update()

      if self.instances:
        self.img_inst_a_vis.set_data(self.scan_a.proj_inst_color[..., ::-1])
        self.img_inst_a_vis.update()
        self.img_inst_b_vis.set_data(self.scan_b.proj_inst_color[..., ::-1])
        self.img_inst_b_vis.update()

