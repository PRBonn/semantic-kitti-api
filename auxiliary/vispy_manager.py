#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas


class VispyManager(object):
  def __init__(self, key_press, draw):
    self.canvas = SceneCanvas(keys='interactive', show=True)
    self.canvas.events.key_press.connect(key_press)
    self.canvas.events.draw.connect(draw)
    self.grid = self.canvas.central_widget.add_grid()
    self.destroyed = False

  def block_key_press(self):
    self.canvas.events.key_press.block()

  def key_press_blocked(self):
    return self.canvas.events.key_press.blocked()

  def key_press_unblocked(self):
    return not self.canvas.events.key_press.blocked()

  def unblock_key_press(self):
    self.canvas.events.key_press.unblock()

  def add_viewbox(self, row, col, border_color='white'):
    view = vispy.scene.widgets.ViewBox(border_color=border_color, parent=self.canvas.scene)
    self.grid.add_widget(view, row, col)
    vis = visuals.Markers()
    view.camera = 'turntable'
    view.add(vis)
    visuals.XYZAxis(parent=view.scene)
    return view, vis

  def destroy(self):
    self.canvas.close()
    vispy.app.quit()
    self.destroyed = True

  def __del__(self):
    if not self.destroyed:
      self.destroy()

  def run(self):
    vispy.app.run()