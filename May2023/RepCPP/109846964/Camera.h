

#pragma once

#include <glm/glm.hpp>


namespace tdogl {


class Camera {
public:
Camera();


const glm::vec3& position() const;
void setPosition(const glm::vec3& position);


float fieldOfView() const;
void setFieldOfView(float fieldOfView);


float nearPlane() const;


float farPlane() const;


void setNearAndFarPlanes(float nearPlane, float farPlane);






void lookAt(glm::vec3 position);


float viewportAspectRatio() const;
void setViewportAspectRatio(float viewportAspectRatio);








glm::mat4 matrix() const;


glm::mat4 projection() const;


glm::mat4 view() const;

private:
glm::vec3 _position;
glm::vec3 _lookAt;
float _fieldOfView;
float _nearPlane;
float _farPlane;
float _viewportAspectRatio;

};

}

