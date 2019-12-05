#version 330 core

layout(points) in;
layout(line_strip, max_vertices = 6) out;

uniform mat4 mvp;
uniform mat4 pose;
uniform float size;

out vec4 color;

void main()
{
    color = vec4(1, 0, 0, 1);
    gl_Position = mvp * pose * vec4(0, 0, 0, 1);
    EmitVertex();
    gl_Position = mvp * pose * vec4(size, 0, 0, 1);
    EmitVertex();
    EndPrimitive();
    
    color = vec4(0, 1, 0, 1);
    gl_Position = mvp * pose * vec4(0, 0, 0, 1);
    EmitVertex();
    gl_Position = mvp * pose *  vec4(0, size, 0, 1);
    EmitVertex();
    EndPrimitive();
    
    color = vec4(0, 0, 1, 1);
    gl_Position = mvp * pose * vec4(0, 0, 0, 1);
    EmitVertex();
    gl_Position = mvp * pose * vec4(0, 0, size, 1);
    EmitVertex();
    EndPrimitive();

}
