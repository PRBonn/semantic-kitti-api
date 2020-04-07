import OpenGL.GL as gl
gl.ERROR_CHECKING = True
gl.ERROR_ON_COPY = True
gl.WARN_ON_FORMAT_UNAVAILABLE = True
import numpy as np
import re
"""
 openGL Object Wrapper (GLOW) in python.

 Some convenience classes to simplify resource management

"""

WARN_INVALID_UNIFORMS = False


def vec2(x, y):
  """ returns an vec2-compatible numpy array """
  return np.array([x, y], dtype=np.float32)


def vec3(x, y, z):
  """ returns an vec3-compatible numpy array """
  return np.array([x, y, z], dtype=np.float32)


def vec4(x, y, z, w):
  """ returns an vec4-compatible numpy array """
  return np.array([x, y, z, w], dtype=np.float32)


def ivec2(x, y):
  """ returns an ivec2-compatible numpy array """
  return np.array([x, y], dtype=np.int32)


def ivec3(x, y, z):
  """ returns an ivec3-compatible numpy array """
  return np.array([x, y, z], dtype=np.int32)


def ivec4(x, y, z, w):
  """ returns an ivec4-compatible numpy array """
  return np.array([x, y, z, w], dtype=np.int32)


def uivec2(x, y):
  """ returns an ivec2-compatible numpy array """
  return np.array([x, y], dtype=np.uint32)


def uivec3(x, y, z):
  """ returns an ivec3-compatible numpy array """
  return np.array([x, y, z], dtype=np.uint32)


def uivec4(x, y, z, w):
  """ returns an ivec4-compatible numpy array """
  return np.array([x, y, z, w], dtype=np.uint32)


class GlBuffer:
  """ 
   Buffer object representing a vertex array buffer.
  """

  def __init__(self, target=gl.GL_ARRAY_BUFFER, usage=gl.GL_STATIC_DRAW):
    self.id_ = gl.glGenBuffers(1)
    self.target_ = target
    self.usage_ = usage

  # def __del__(self):
  #   gl.glDeleteBuffers(1, self.id_)

  def assign(self, array):
    gl.glBindBuffer(self.target_, self.id_)
    gl.glBufferData(self.target_, array, self.usage_)
    gl.glBindBuffer(self.target_, 0)

  def bind(self):
    gl.glBindBuffer(self.target_, self.id_)

  def release(self):
    gl.glBindBuffer(self.target_, 0)

  @property
  def id(self):
    return self.id_

  @property
  def usage(self):
    return self.usage_

  @property
  def target(self):
    return self.target_


class GlTextureRectangle:
  def __init__(self, width, height, internalFormat=gl.GL_RGBA, format=gl.GL_RGBA):
    self.id_ = gl.glGenTextures(1)
    self.internalFormat_ = internalFormat  # gl.GL_RGB_FLOAT, gl.GL_RGB_UNSIGNED, ...
    self.format = format  # GL_RG. GL_RG_INTEGER, ...

    self.width_ = width
    self.height_ = height

    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, self.id_)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
    gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, 0)

  def bind(self, textureUnitId):
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(textureUnitId))
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, self.id_)

  def release(self, textureUnitId):
    gl.glActiveTexture(gl.GL_TEXTURE0 + int(textureUnitId))
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, 0)

  def assign(self, array):
    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, self.id_)

    if array.dtype == np.uint8:
      gl.glTexImage2D(gl.GL_TEXTURE_RECTANGLE, 0, self.internalFormat_, self.width_, self.height_, 0, self.format,
                      gl.GL_UNSIGNED_BYTE, array)
    elif array.dtype == np.float32:
      gl.glTexImage2D(gl.GL_TEXTURE_RECTANGLE, 0, self.internalFormat_, self.width_, self.height_, 0, self.format,
                      gl.GL_FLOAT, array)
    else:
      raise NotImplementedError("pixel type not implemented.")

    gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, 0)

  @property
  def id(self):
    return self.id_


class GlShader:
  def __init__(self, shader_type, source):
    self.code_ = source
    self.shader_type_ = shader_type

    self.id_ = gl.glCreateShader(self.shader_type_)
    gl.glShaderSource(self.id_, source)

    gl.glCompileShader(self.id_)

    success = gl.glGetShaderiv(self.id_, gl.GL_COMPILE_STATUS)
    if success == gl.GL_FALSE:
      error_string = gl.glGetShaderInfoLog(self.id_).decode("utf-8")
      raise RuntimeError(error_string)

  def __del__(self):
    gl.glDeleteShader(self.id_)

  @property
  def type(self):
    return self.shader_type_

  @property
  def id(self):
    return self.id_

  @property
  def code(self):
    return self.code_

  @staticmethod
  def fromFile(shader_type, filename):
    f = open(filename)
    source = "\n".join(f.readlines())
    # todo: preprocess.
    f.close()

    return GlShader(shader_type, source)


class GlProgram:
  """ An OpenGL program handle. """

  def __init__(self):
    self.id_ = gl.glCreateProgram()
    self.shaders_ = {}
    self.uniform_types_ = {}
    self.is_linked = False

  def __del__(self):
    gl.glDeleteProgram(self.id_)

  def bind(self):
    if not self.is_linked:
      raise RuntimeError("Program must be linked before usage.")
    gl.glUseProgram(self.id_)

  def release(self):
    gl.glUseProgram(0)

  def attach(self, shader):
    self.shaders_[shader.type] = shader

  def __setitem__(self, name, value):
    # quitely ignore
    if name not in self.uniform_types_:
      if WARN_INVALID_UNIFORMS: print("No uniform with name '{}' available.".format(name))
      return

    loc = gl.glGetUniformLocation(self.id_, name)
    T = self.uniform_types_[name]

    if T == "int":
      gl.glUniform1i(loc, np.int32(value))
    if T == "uint":
      gl.glUniform1ui(loc, np.uint32(value))
    elif T == "float":
      gl.glUniform1f(loc, np.float32(value))
    elif T == "bool":
      gl.glUniform1f(loc, value)
    elif T == "vec2":
      gl.glUniform2fv(loc, 1, value)
    elif T == "vec3":
      gl.glUniform3fv(loc, 1, value)
    elif T == "vec4":
      gl.glUniform4fv(loc, 1, value)
    elif T == "ivec2":
      gl.glUniform2iv(loc, 1, value)
    elif T == "ivec3":
      gl.glUniform3iv(loc, 1, value)
    elif T == "ivec4":
      gl.glUniform4iv(loc, 1, value)
    elif T == "uivec2":
      gl.glUniform2uiv(loc, 1, value)
    elif T == "uivec3":
      gl.glUniform3uiv(loc, 1, value)
    elif T == "uivec4":
      gl.glUniform4uiv(loc, 1, value)
    elif T == "mat4":
      #print("set matrix: ", value)
      gl.glUniformMatrix4fv(loc, 1, False, value.astype(np.float32))
    elif T == "sampler2D":
      gl.glUniform1i(loc, np.int32(value))
    elif T == "sampler2DRect":
      gl.glUniform1i(loc, np.int32(value))
    else:
      raise NotImplementedError("uniform type {} not implemented. :(".format(T))

  def link(self):
    if gl.GL_VERTEX_SHADER not in self.shaders_ or gl.GL_FRAGMENT_SHADER not in self.shaders_:
      raise RuntimeError("program needs at least vertex and fragment shader")

    for shader in self.shaders_.values():
      gl.glAttachShader(self.id_, shader.id)
      for line in shader.code.split("\n"):
        match = re.search(r"uniform\s+(\S+)\s+(\S+)\s*;", line)
        if match:
          self.uniform_types_[match.group(2)] = match.group(1)

    gl.glLinkProgram(self.id_)
    isLinked = bool(gl.glGetProgramiv(self.id_, gl.GL_LINK_STATUS))
    if not isLinked:
      msg = gl.glGetProgramInfoLog(self.id_)

      raise RuntimeError(str(msg.decode("utf-8")))

    # after linking we don't need the source code anymore.
    self.shaders_ = {}
    self.is_linked = True
