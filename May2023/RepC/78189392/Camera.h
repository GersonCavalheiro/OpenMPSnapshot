#pragma once
#include <glm\glm.hpp>
#include <glm/gtx/transform.hpp>
#include "Transofrm.h"
#include <iostream>
#include <iomanip>
class Camera
{
public:
Camera(float aspect_ratio, float fov, float near_clipping, float far_clipping);
void SetPosition(glm::vec3 pos);
void SetPosition(float x, float y, float z);
void Move(glm::vec3 dpos);
void Move(float dx, float dy, float dz);
void SetRotationVector(glm::vec3 direction, glm::vec3 up);
void SetRotationAngles(glm::vec3 angles);
void SetRotationAngles(float yaw, float pitch, float roll);
void Rotate(glm::vec3 d_rot);
void Rotate(float d_yaw, float d_pitch, float d_roll);
Transform GetTransform();
private:
glm::mat4 camera_matrix, projection_matrix;
glm::vec3 position, direction, up;
glm::vec3 angles;
void calc_camera();
};