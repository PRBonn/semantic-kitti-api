#!/usr/bin/python3

import math
import sys
import os
import time
import argparse

import glfw

import OpenGL
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import *

import numpy as np
import yaml

import imgui
from imgui.integrations.glfw import GlfwRenderer

import auxiliary.glow as glow
from auxiliary.camera import Camera

OpenGL.ERROR_ON_COPY = True
OpenGL.ERROR_CHECKING = True


def glPerspective(fov, aspect, znear, zfar):
  # https://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
  # assert(znear > 0.0)
  M = np.zeros((4, 4))

  # Copied from gluPerspective
  f = 1.0 / math.tan(0.5 * fov)

  M[0, 0] = f / aspect
  M[1, 1] = f
  M[2, 2] = (znear + zfar) / (znear - zfar)
  M[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
  M[3, 2] = -1.0

  return M


def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed


class Window:
  def __init__(self):
    if not glfw.init():
      raise RuntimeError("Unable to initialize glfw.")

    w_window, h_window = 800, 600

    self.window = glfw.create_window(w_window, h_window, "Voxel Visualizer", None, None)

    if not self.window:
      glfw.terminate()
      raise RuntimeError("Unable to create window.")

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    w_mon, h_mon = mode.size
    # center window.
    glfw.set_window_pos(self.window, int(0.5 * w_mon - 0.5 * w_window), int(0.5 * h_mon - 0.5 * h_window))

    # Make the window's context current
    glfw.make_context_current(self.window)

    glfw.set_framebuffer_size_callback(self.window, self.on_resize)
    # add mouse handlers.
    glfw.set_input_mode(self.window, glfw.STICKY_MOUSE_BUTTONS, glfw.TRUE)
    glfw.set_mouse_button_callback(self.window, self.on_mouse_btn)
    glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
    glfw.set_window_size_callback(self.window, self.on_resize)

    glfw.set_key_callback(self.window, self.keyboard_callback)
    glfw.set_char_callback(self.window, self.char_callback)
    glfw.set_scroll_callback(self.window, self.scroll_callback)

    self.voxel_dims = glow.ivec3(256, 256, 32)

    # read config file.
    CFG = yaml.safe_load(open("config/semantic-kitti.yaml", 'r'))
    color_dict = CFG["color_map"]
    self.label_colors = glow.GlTextureRectangle(1024, 1, internalFormat=GL_RGB, format=GL_RGB)

    cols = np.zeros((1024 * 3), dtype=np.uint8)
    for label_id, color in color_dict.items():
      cols[3 * label_id + 0] = color[2]
      cols[3 * label_id + 1] = color[1]
      cols[3 * label_id + 2] = color[0]

    self.label_colors.assign(cols)

    self.initializeGL()

    # initialize imgui
    imgui.create_context()
    self.impl = GlfwRenderer(self.window, attach_callbacks=False)

    self.on_resize(self.window, w_window, h_window)

    self.data = []
    self.isDrag = False
    self.buttonPressed = None
    self.cam = Camera()
    self.cam.lookAt(25.0, 25.0, 25.0, 0.0, 0.0, 0.0)

    self.currentTimestep = 0
    self.sliderValue = 0
    self.showLabels = True

  def initializeGL(self):
    """ initialize GL related stuff. """

    self.num_instances = np.prod(self.voxel_dims)

    # see https://stackoverflow.com/questions/28375338/cube-using-single-gl-triangle-strip, but the normals are a problem.
    # verts = np.array([
    #     -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1,
    #     -1, 1, -1, -1, -1, 1, -1, 1, 1, -1
    # ],
    #                  dtype=np.float32)

    # yapf: disable

    p1 = [1, 0, 0]
    p2 = [0, 0, 0]
    p3 = [1, 1, 0]
    p4 = [0, 1, 0]
    p5 = [1, 0, 1]
    p6 = [0, 0, 1]
    p7 = [0, 1, 1]
    p8 = [1, 1, 1]

    verts = np.array([
        # first face
        p4, p3, p7, p3, p7, p8,
        # second face
        p7, p8, p5, p7, p6, p5,
        # third face
        p8, p5, p3, p5, p3, p1,
        # fourth face
        p3, p1, p4, p1, p4, p2,
        # fifth face
        p4, p2, p7, p2, p7, p6,
        # sixth face
        p6, p5, p2, p5, p2, p1
    ], dtype=np.float32).reshape(-1)

    normals = np.array([[0, 1, 0] * 6,
                        [0, 0, 1] * 6,
                        [1, 0, 0] * 6,
                        [0, 0, -1] * 6,
                        [-1, 0, 0] * 6,
                        [0, -1, 0] * 6
                        ], dtype=np.float32).reshape(-1)

    # yapf: enable
    glow.WARN_INVALID_UNIFORMS = True

    self.labels = np.array([], dtype=np.float32)

    self.cube_verts = glow.GlBuffer()
    self.cube_verts.assign(verts)
    self.cube_normals = glow.GlBuffer()
    self.cube_normals.assign(normals)
    self.label_vbo = glow.GlBuffer()

    glPointSize(5.0)

    self.vao = glGenVertexArrays(1)
    glBindVertexArray(self.vao)

    SIZEOF_FLOAT = 4

    self.cube_verts.bind()
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * SIZEOF_FLOAT, GLvoidp(0))
    glEnableVertexAttribArray(0)
    self.cube_verts.release()

    self.cube_normals.bind()
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * SIZEOF_FLOAT, GLvoidp(0))
    glEnableVertexAttribArray(1)
    self.cube_normals.release()

    glEnableVertexAttribArray(2)
    self.label_vbo.bind()
    # Note: GL_UNSINGED_INT did not work as expected! I could not figure out what was wrong there!
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, SIZEOF_FLOAT, GLvoidp(0))
    self.label_vbo.release()
    glVertexAttribDivisor(2, 1)

    glBindVertexArray(0)

    self.program = glow.GlProgram()
    self.program.attach(glow.GlShader.fromFile(GL_VERTEX_SHADER, "auxiliary/shaders/draw_voxels.vert"))
    self.program.attach(glow.GlShader.fromFile(GL_FRAGMENT_SHADER, "auxiliary/shaders/draw_voxels.frag"))
    self.program.link()

    self.prgDrawPose = glow.GlProgram()
    self.prgDrawPose.attach(glow.GlShader.fromFile(GL_VERTEX_SHADER, "auxiliary/shaders/empty.vert"))
    self.prgDrawPose.attach(glow.GlShader.fromFile(GL_GEOMETRY_SHADER, "auxiliary/shaders/draw_pose.geom"))
    self.prgDrawPose.attach(glow.GlShader.fromFile(GL_FRAGMENT_SHADER, "auxiliary/shaders/passthrough.frag"))
    self.prgDrawPose.link()

    self.prgTestUniform = glow.GlProgram()
    self.prgTestUniform.attach(glow.GlShader.fromFile(GL_VERTEX_SHADER, "auxiliary/shaders/check_uniforms.vert"))
    self.prgTestUniform.attach(glow.GlShader.fromFile(GL_FRAGMENT_SHADER, "auxiliary/shaders/passthrough.frag"))
    self.prgTestUniform.link()

    # general parameters
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_LINE_SMOOTH)

    # x = forward, y = left, z = up to x = right, y = up, z = backward
    self.conversion_ = np.array([0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32).reshape(4, 4)

    self.program.bind()

    self.program["voxel_size"] = 0.5
    self.program["voxel_dims"] = self.voxel_dims
    self.program["label_colors"] = 0

    self.program.release()

    self.vao_no_points = glGenVertexArrays(1)

  def open_directory(self, directory):
    """ open given sequences directory and get filenames of relevant files. """
    self.subdirs = [subdir for subdir in ["voxels", "predictions"] if os.path.exists(os.path.join(directory, subdir))]

    if len(self.subdirs) == 0: raise RuntimeError("Neither 'voxels' nor 'predictions' found in " + directory)

    self.availableData = {}
    self.data = {}

    for subdir in self.subdirs:
      self.availableData[subdir] = []
      self.data[subdir] = {}
      complete_path = os.path.join(directory, subdir)
      files = os.listdir(complete_path)

      data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".bin")])
      if len(data) > 0:
        self.availableData[subdir].append("input")
        self.data[subdir]["input"] = data
        self.num_scans = len(data)

      data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".label")])
      if len(data) > 0:
        self.availableData[subdir].append("labels")
        self.data[subdir]["labels"] = data
        self.num_scans = len(data)

      data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".invalid")])
      if len(data) > 0:
        self.availableData[subdir].append("invalid")
        self.data[subdir]["invalid"] = data
        self.num_scans = len(data)

      data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".occluded")])
      if len(data) > 0:
        self.availableData[subdir].append("occluded")
        self.data[subdir]["occluded"] = data
        self.num_scans = len(data)

    self.current_subdir = 0
    self.current_data = self.availableData[self.subdirs[self.current_subdir]][0]

    self.currentTimestep = 0
    self.sliderValue = 0

    self.lastChange = None
    self.lastUpdate = time.time()
    self.button_backward_hold = False
    self.button_forward_hold = False

    # todo: modify based on available stuff.
    self.showLabels = (self.current_data == "labels")
    self.showInput = (self.current_data == "input")
    self.showInvalid = (self.current_data == "invalid")
    self.showOccluded = (self.current_data == "occludded")

  def setCurrentBufferData(self, data_name, t):
    # update buffer content with given data identified by data_name.
    subdir = self.subdirs[self.current_subdir]

    if len(self.data[subdir][data_name]) < t: return False

    # Note: uint with np.uint32 did not work as expected! (with instances and uint32 this causes problems!)
    if data_name == "labels":
      buffer_data = np.fromfile(self.data[subdir][data_name][t], dtype=np.uint16).astype(np.float32)
    else:
      buffer_data = unpack(np.fromfile(self.data[subdir][data_name][t], dtype=np.uint8)).astype(np.float32)

    self.label_vbo.assign(buffer_data)

    return True

  def on_resize(self, window, w, h):
    # set projection matrix
    fov = math.radians(45.0)
    aspect = w / h

    self.projection_ = glPerspective(fov, aspect, 0.1, 2000.0)

    self.impl.resize_callback(window, w, h)

  def on_mouse_btn(self, window, button, action, mods):
    x, y = glfw.get_cursor_pos(self.window)

    imgui.get_io().mouse_pos = (x, y)
    if imgui.get_io().want_capture_mouse:

      return

    if action == glfw.PRESS:
      self.buttonPressed = button
      self.isDrag = True
      self.cam.mousePressed(x, y, self.buttonPressed, None)
    else:
      self.buttonPressed = None
      self.isDrag = False
      self.cam.mouseReleased(x, y, self.buttonPressed, None)

  def on_mouse_move(self, window, x, y):
    if self.isDrag:
      self.cam.mouseMoved(x, y, self.buttonPressed, None)

  def keyboard_callback(self, window, key, scancode, action, mods):
    self.impl.keyboard_callback(window, key, scancode, action, mods)

    if not imgui.get_io().want_capture_keyboard:
      if key == glfw.KEY_B or key == glfw.KEY_LEFT:
        self.currentTimestep = self.sliderValue = max(0, self.currentTimestep - 1)

      if key == glfw.KEY_N or key == glfw.KEY_RIGHT:
        self.currentTimestep = self.sliderValue = min(self.num_scans - 1, self.currentTimestep + 1)

      if key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
        exit(0)

  def char_callback(self, window, char):
    self.impl.char_callback(window, char)

  def scroll_callback(self, window, x_offset, y_offset):
    self.impl.scroll_callback(window, x_offset, y_offset)

  def run(self):
    # Loop until the user closes the window
    while not glfw.window_should_close(self.window):
      # Poll for and process events
      glfw.poll_events()

      # build gui.
      self.impl.process_inputs()

      w, h = glfw.get_window_size(self.window)
      glViewport(0, 0, w, h)

      imgui.new_frame()

      timeline_height = 35
      imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0)
      imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 0)

      imgui.set_next_window_position(0, h - timeline_height - 10)
      imgui.set_next_window_size(w, timeline_height)

      imgui.begin("Timeline", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
      imgui.columns(1)
      imgui.same_line(0, 0)
      imgui.push_item_width(-50)
      changed, value = imgui.slider_int("", self.sliderValue, 0, self.num_scans - 1)
      if changed: self.sliderValue = value
      if self.sliderValue != self.currentTimestep:
        self.currentTimestep = self.sliderValue

      imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 3)

      play_delay = 1
      refresh_rate = 0.05

      current_time = time.time()

      imgui.same_line(spacing=5)
      changed = imgui.button("<", 20)
      if self.currentTimestep > 0:
        # just a click
        if changed:
          self.currentTimestep = self.sliderValue = self.currentTimestep - 1
          self.lastUpdate = current_time

        # button pressed.
        if imgui.is_item_active() and not self.button_backward_hold:
          self.hold_start = current_time
          self.button_backward_hold = True

        if not imgui.is_item_active() and self.button_backward_hold:
          self.button_backward_hold = False

        # start playback when button pressed long enough
        if self.button_backward_hold and ((current_time - self.hold_start) > play_delay):
          if (current_time - self.lastUpdate) > refresh_rate:
            self.currentTimestep = self.sliderValue = self.currentTimestep - 1
            self.lastUpdate = current_time

      imgui.same_line(spacing=2)
      changed = imgui.button(">", 20)

      if self.currentTimestep < self.num_scans - 1:
        # just a click
        if changed:
          self.currentTimestep = self.sliderValue = self.currentTimestep + 1
          self.lastUpdate = current_time

        # button pressed.
        if imgui.is_item_active() and not self.button_forward_hold:
          self.hold_start = current_time
          self.button_forward_hold = True

        if not imgui.is_item_active() and self.button_forward_hold:
          self.button_forward_hold = False

        # start playback when button pressed long enough
        if self.button_forward_hold and ((current_time - self.hold_start) > play_delay):
          if (current_time - self.lastUpdate) > refresh_rate:
            self.currentTimestep = self.sliderValue = self.currentTimestep + 1
            self.lastUpdate = current_time

      imgui.pop_style_var(3)
      imgui.end()

      imgui.set_next_window_position(20, 20, imgui.FIRST_USE_EVER)
      imgui.set_next_window_size(200, 150, imgui.FIRST_USE_EVER)
      imgui.begin("Show Data")

      if len(self.subdirs) > 1:
        for i, subdir in enumerate(self.subdirs):
          changed, value = imgui.checkbox(subdir, self.current_subdir == i)
          if i < len(self.subdirs) - 1: imgui.same_line()
          if changed and value: self.current_subdir = i

      subdir = self.subdirs[self.current_subdir]

      data_available = "input" in self.availableData[subdir]
      if data_available:
        imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
      else:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

      changed, value = imgui.checkbox("input", self.showInput)
      if changed and value and data_available:
        self.showInput = True
        self.showLabels = False

      imgui.pop_style_var()

      data_available = "labels" in self.availableData[subdir]
      if data_available:
        imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
      else:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

      changed, value = imgui.checkbox("labels", self.showLabels)
      if changed and value and data_available:
        self.showInput = False
        self.showLabels = True

      imgui.pop_style_var()

      data_available = "invalid" in self.availableData[subdir]
      if data_available:
        imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
      else:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

      changed, value = imgui.checkbox("invalid", self.showInvalid)
      if changed and data_available: self.showInvalid = value

      imgui.pop_style_var()

      data_available = "occluded" in self.availableData[subdir]
      if data_available:
        imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
      else:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

      changed, value = imgui.checkbox("occluded", self.showOccluded)
      if changed and data_available: self.showOccluded = value

      imgui.pop_style_var()

      imgui.end()

      # imgui.show_demo_window()

      showData = []
      if self.showInput: showData.append("input")
      if self.showOccluded: showData.append("occluded")
      if self.showInvalid: showData.append("invalid")

      mvp = self.projection_ @ self.cam.matrix @ self.conversion_

      glClearColor(1.0, 1.0, 1.0, 1.0)

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      glBindVertexArray(self.vao)

      self.program.bind()
      self.program["mvp"] = mvp.transpose()
      self.program["view_mat"] = (self.cam.matrix @ self.conversion_).transpose()
      self.program["lightPos"] = glow.vec3(10, 10, 10)
      self.program["voxel_scale"] = 0.8
      self.program["voxel_alpha"] = 1.0
      self.program["use_label_colors"] = True

      self.label_colors.bind(0)

      if self.showLabels:
        self.setCurrentBufferData("labels", self.currentTimestep)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)

      self.program["use_label_colors"] = False
      self.program["voxel_color"] = glow.vec3(0.3, 0.3, 0.3)

      self.program["voxel_alpha"] = 0.5

      for data_name in showData:
        self.program["voxel_scale"] = 0.5
        if data_name == "input": self.program["voxel_scale"] = 0.8

        self.setCurrentBufferData(data_name, self.currentTimestep)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)

      self.program.release()
      self.label_colors.release(0)

      glBindVertexArray(self.vao_no_points)

      self.prgDrawPose.bind()
      self.prgDrawPose["mvp"] = mvp.transpose()
      self.prgDrawPose["pose"] = np.identity(4, dtype=np.float32)
      self.prgDrawPose["size"] = 1.0

      glDrawArrays(GL_POINTS, 0, 1)
      self.prgDrawPose.release()

      glBindVertexArray(0)

      # draw gui ontop.
      imgui.render()
      self.impl.render(imgui.get_draw_data())

      # Swap front and back buffers
      glfw.swap_buffers(self.window)


if __name__ == "__main__":

  parser = argparse.ArgumentParser("./visualize_voxels.py")
  parser.add_argument(
      '--dataset',
      '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )

  parser.add_argument(
      '--sequence',
      '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )

  FLAGS, unparsed = parser.parse_known_args()

  sequence_directory = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence)

  window = Window()
  window.open_directory(sequence_directory)

  window.run()

  glfw.terminate()
