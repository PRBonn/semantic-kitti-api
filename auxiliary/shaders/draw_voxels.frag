#version 330 core

// simple Blinn-Phong Shading.

out vec4 out_color;

in vec4 color;
in vec3 position;
in vec3 normal;

uniform mat4 view_mat;
uniform vec3 lightPos;

void main()
{
  vec3 viewPos = view_mat[3].xyz;

  vec3 ambient = 0.05 * color.xyz;
  
  vec3 lightDir = normalize(lightPos - position);
  vec3 normal1 = normalize(normal);
  float diff = max(dot(lightDir, normal1), 0.0);
  vec3 diffuse = diff * color.xyz;
  
  vec3 viewDir = normalize(viewPos - position);
  vec3 reflectDir = reflect(-lightDir, normal);
  vec3 halfwayDir = normalize(lightDir + viewDir);
  
  float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
  vec3 specular = vec3(0.1) * spec;
  
  out_color = vec4(ambient + diffuse + specular, 1.0);
}
