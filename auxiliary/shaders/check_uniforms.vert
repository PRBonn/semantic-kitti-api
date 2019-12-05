
#version 330 core


uniform mat4 test;

out vec4 color;

void main() {

  float value = float(gl_VertexID);

  gl_Position = vec4(value/16.0, value/16.0, 0, 1);
  if(test[int(value/4)][int(value)%4] == value)
  { 
    color = vec4(0,1,0,1);
  }
  else
  {
    color = vec4(1,0,0,1);
  }
}