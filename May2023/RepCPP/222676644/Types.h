#pragma once

#include "glm/glm.hpp"
#include "vector"

struct SRay
{
glm::vec3 m_start;
glm::vec3 m_dir;
float length;

SRay(glm::vec3 startPos, glm::vec3 dir) : m_start(startPos), m_dir(dir), length(0) {}
};

struct SCamera
{
glm::vec3 m_pos;          
glm::vec3 m_forward;      
glm::vec3 m_up;
glm::vec3 m_right;

glm::vec2 m_FOV;    
glm::uvec2 m_resolution;  

std::vector<glm::vec3> m_pixels;  
};

struct SMesh
{
std::vector<glm::vec3> m_vertices;  
std::vector<glm::vec2> m_textures;
std::vector<glm::vec3> m_normals;
std::vector<glm::uvec3> m_triangles;  
std::vector<unsigned int> m_triangles_normals;
};