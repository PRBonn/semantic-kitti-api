# version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in float in_label; // Note: uint with np.uint32 did not work as expected! 

uniform mat4 mvp;
uniform mat4 view_mat;
uniform sampler2DRect label_colors;
uniform bool use_label_colors;

uniform ivec3 voxel_dims;
uniform float voxel_size;
uniform float voxel_scale;
uniform vec3 voxel_color;
uniform float voxel_alpha;

out vec3 position;
out vec3 normal;
out vec4 color;


void main()
{
  // instance id corresponds to the index in the grid.
  vec3 idx;
  idx.x = int(float(gl_InstanceID) / float(voxel_dims.y * voxel_dims.z));
  idx.y = int(float(gl_InstanceID - idx.x * voxel_dims.y * voxel_dims.z) / float(voxel_dims.z));
  idx.z = int(gl_InstanceID - idx.x * voxel_dims.y * voxel_dims.z - idx.y * voxel_dims.z);

  // centerize the voxelgrid.
  vec3 offset = voxel_size * vec3(0, 0.5, 0.5) * voxel_dims;
  vec3 pos = voxel_scale * voxel_size * (in_position - 0.5); // centerize the voxel coordinates and resize.

  position = (view_mat * vec4(pos + idx * voxel_size - offset, 1)).xyz;
  normal = (view_mat * vec4(in_normal, 0)).xyz;

  uint label = uint(in_label);

  if(label == uint(0)) // empty voxels
    gl_Position = vec4(-10, -10, -10, 1);
  else
    gl_Position = mvp * vec4(pos + idx * voxel_size - offset, 1);
  
  color = vec4(voxel_color, voxel_alpha);
  if (use_label_colors) color =  vec4(texture(label_colors, vec2(label, 0)).rgb, voxel_alpha);
}