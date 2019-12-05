import math
import numpy as np
import time
import glfw

def RotX(angle):
  sin_t = math.sin(angle)
  cos_t = math.cos(angle)

  return np.array([1, 0, 0, 0, 0, cos_t, -sin_t, 0, 0, sin_t, cos_t, 0, 0, 0, 0, 1], dtype=np.float32).reshape(4, 4)


def RotY(angle):
  sin_t = math.sin(angle)
  cos_t = math.cos(angle)

  return np.array([cos_t, 0, sin_t, 0, 0, 1, 0, 0, -sin_t, 0, cos_t, 0, 0, 0, 0, 1], dtype=np.float32).reshape(4, 4)


def Trans(x, y, z):
  return np.array([1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1], dtype=np.float32).reshape(4, 4)


class Camera:
  """ Camera for handling the view matrix based on mouse inputs. """

  def __init__(self):
    self.x_ = self.y_ = self.z_ = 0.0
    self.pitch_ = 0.0
    self.yaw_ = 0.0

    self.startdrag_ = False
    self.startTime_ = 0
    self.startx_ = 0
    self.starty_ = 0
    self.startyaw_ = 0
    self.startpitch_ = 0

    self.forwardVel_ = 0.0
    self.upVel_ = 0.0
    self.sideVel_ = 0.0
    self.turnVel_ = 0.0
    self.startdrag_ = False

  def lookAt(self, x_cam, y_cam, z_cam, x_ref, y_ref, z_ref):
    self.x_ = x_cam
    self.y_ = y_cam
    self.z_ = z_cam

    x = x_ref - self.x_
    y = y_ref - self.y_
    z = z_ref - self.z_
    length = math.sqrt(x * x + y * y + z * z)

    self.pitch_ = math.asin(y / length)  # = std: : acos(-dir.y()) - M_PI_2 in [-pi/2, pi/2]
    self.yaw_ = math.atan2(-x, -z)

    self.startdrag_ = False

  @property
  def matrix(self):
    # current time.
    end = time.time()
    dt = end - self.startTime_

    if dt > 0 and self.startdrag_:
      # apply velocity & reset timer...
      self.rotate(self.turnVel_ * dt, 0.0)
      self.translate(self.forwardVel_ * dt, self.upVel_ * dt, self.sideVel_ * dt)
      self.startTime_ = end

    # recompute the view matrix (Euler angles) Remember: Inv(AB) = Inv(B)*Inv(A)
    # Inv(translate*rotateYaw*rotatePitch) = Inv(rotatePitch)*Inv(rotateYaw)*Inv(translate)
    view_ = RotX(-self.pitch_)
    view_ = view_ @ RotY(-self.yaw_)
    view_ = view_ @ Trans(-self.x_, -self.y_, -self.z_)

    return view_

  def mousePressed(self, x, y, btn, modifier):
    self.startx_ = x
    self.starty_ = y
    self.startyaw_ = self.yaw_
    self.startpitch_ = self.pitch_
    self.startTime_ = time.time()
    self.startdrag_ = True

    return True

  def mouseReleased(self, x, y, btn, modifier):
    self.forwardVel_ = 0.0
    self.upVel_ = 0.0
    self.sideVel_ = 0.0
    self.turnVel_ = 0.0
    self.startdrag_ = False

    return True

  def translate(self, forward, up, sideways):
    # forward = -z, sideways = x , up = y. Remember: inverse of yaw is applied, i.e., we have to apply yaw (?)
    # Also keep in mind: sin(-alpha) = -sin(alpha) and cos(-alpha) = -cos(alpha)
    # We only apply the yaw to move along the yaw direction;
    #  x' = x*cos(yaw) - z*sin(yaw)
    #  z' = x*sin(yaw) + z*cos(yaw)
    s = math.sin(self.yaw_)
    c = math.cos(self.yaw_)

    self.x_ = self.x_ + sideways * c - forward * s
    self.y_ = self.y_ + up
    self.z_ = self.z_ - (sideways * s + forward * c)

  def rotate(self, yaw, pitch):
    self.yaw_ += yaw
    self.pitch_ += pitch
    if self.pitch_ < -0.5 * math.pi:
      self.pitch_ = -0.5 * math.pi
    if self.pitch_ > 0.5 * math.pi:
      self.pitch_ = 0.5 * math.pi

  def mouseMoved(self, x, y, btn, modifier):
    # some constants.
    MIN_MOVE = 0
    WALK_SENSITIVITY = 0.5
    TURN_SENSITIVITY = 0.01
    SLIDE_SENSITIVITY = 0.5
    RAISE_SENSITIVITY = 0.5

    LOOK_SENSITIVITY = 0.01
    FREE_TURN_SENSITIVITY = 0.01

    dx = x - self.startx_
    dy = y - self.starty_

    if dx > 0.0:
      dx = max(0.0, dx - MIN_MOVE)
    if dx < 0.0:
      dx = min(0.0, dx + MIN_MOVE)
    if dy > 0.0:
      dy = max(0.0, dy - MIN_MOVE)
    if dy < 0.0:
      dy = min(0.0, dy + MIN_MOVE)

    # idea: if the velocity changes, we have to reset the start_time and update the camera parameters.

    if btn == glfw.MOUSE_BUTTON_RIGHT:

      self.forwardVel_ = 0
      self.upVel_ = 0
      self.sideVel_ = 0
      self.turnVel_ = 0

      self.yaw_ = self.startyaw_ - FREE_TURN_SENSITIVITY * dx
      self.pitch_ = self.startpitch_ - LOOK_SENSITIVITY * dy

      # ensure valid values.
      if self.pitch_ < -0.5 * math.pi:
        self.pitch_ = -0.5 * math.pi
      if self.pitch_ > 0.5 * math.pi:
        self.pitch_ = 0.5 * math.pi

    elif btn == glfw.MOUSE_BUTTON_LEFT:

      # apply transformation:
      end = time.time()
      dt = end - self.startTime_

      if dt > 0.0:
        self.rotate(self.turnVel_ * dt, 0.0)
        self.translate(self.forwardVel_ * dt, self.upVel_ * dt, self.sideVel_ * dt)

        self.startTime_ = end
        # reset timer.

      self.forwardVel_ = -WALK_SENSITIVITY * dy
      self.upVel_ = 0
      self.sideVel_ = 0
      self.turnVel_ = -(TURN_SENSITIVITY * dx)
    elif btn == glfw.MOUSE_BUTTON_MIDDLE:

      # apply transformation:
      end = time.time()
      dt = end - self.startTime_

      if dt > 0.0:
        self.rotate(self.turnVel_ * dt, 0.0)
        self.translate(self.forwardVel_ * dt, self.upVel_ * dt, self.sideVel_ * dt)

        self.startTime_ = end
        # reset timer.

      self.forwardVel_ = 0
      self.upVel_ = -RAISE_SENSITIVITY * dy
      self.sideVel_ = SLIDE_SENSITIVITY * dx
      self.turnVel_ = 0

    return True
