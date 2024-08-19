#pragma once
#include "Math.hpp"
#include "Ray.hpp"


class Camera {
public:
Camera();

void lookAt(const Vec3& target);
void lookAtFrom(const Vec3& target, const Vec3& position);

Vec3 viewportToWorld(const Vec3& point) const;
Vec3 worldToViewport(const Vec3& point) const;
bool isPointInFrustum(const Vec3& point) const;
Ray viewportPointToRay(const Vec3& point) const;

Vec3 position()               const;
Vec3 rightDir()               const;
Vec3 upDir()                  const;
Vec3 forwardDir()             const;
float nearClip()              const;
float farClip()               const;
float aspectRatio()           const;
float horizontalFieldOfView() const;
float verticalFieldOfView()   const;
Vec2 viewportSize()           const;

void setPosition(const Vec3& eyePosition);
void setOrientation(const Vec3& rightDir, const Vec3& upDir, const Vec3& forwardDir);
void setNearClip(float nearClip);
void setFarClip(float farClip);
void setAspectRatio(float aspectRatio);
void setFieldOfView(float horizontalDegrees);
void overrideViewportSize(const Vec2& viewportSize, float nearZ, float farZ);

private:
Vec3 eyePosition_;
Vec3 rightDir_;
Vec3 upDir_;
Vec3 forwardDir_;

float nearClip_;
float farClip_;
float aspectRatio_;
Vec2 fieldOfView_;
Vec2 viewportSize_;

static constexpr Vec3 DEFAULT_POSITION    = Vec3::zero();
static constexpr Vec3 DEFAULT_RIGHT_DIR   = Vec3::right();
static constexpr Vec3 DEFAULT_UP_DIR      = Vec3::up();
static constexpr Vec3 DEFAULT_FORWARD_DIR = Vec3::ahead();

static constexpr float DEFAULT_ASPECT_RATIO  = 4.0f / 3.0f;
static constexpr float DEFAULT_FIELD_OF_VIEW =      60.00f;
static constexpr float DEFAULT_DISTANCE      =    1000.00f;
static constexpr float DEFAULT_NEAR_CLIP     =       0.25f;
static constexpr float DEFAULT_FAR_CLIP      =    1000.00f;

static constexpr float MIN_FIELD_OF_VIEW =   0.00f;
static constexpr float MAX_FIELD_OF_VIEW = 180.00f;

static Vec2 computeSizeFromHorizontalFov(float horizontalFieldOfView, float distanceToPlane, float aspectRatio);
void setupPerspectiveFromSize(const Vec2& viewportSize, float nearZ, float farZ);
};
std::ostream& operator<<(std::ostream& os, const Camera& camera);
