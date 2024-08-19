#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
class Transform
{
public:
Transform();
Transform(glm::mat4 camera_matrix, glm::mat4 projection_matrix);
void Scale(float x, float y, float z);
void Move(float x, float y, float z);
void Rotate(float x, float y, float z);
glm::mat4x4 GetMat();
private:
glm::vec3 scale, rot, move;
glm::mat4 camera_matrix, projection_matrix;
bool is_matrix_set = false;
};
