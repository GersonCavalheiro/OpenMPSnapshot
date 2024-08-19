#pragma once
#include "Math.hpp"
#include "Color.hpp"


class ILight {
protected:
Vec3 position_{};
Color intensity_{};

public:
virtual ~ILight() = default;
virtual Color computeIntensityAtPoint(const Vec3& point) const = 0;
virtual std::string description() const = 0;

constexpr const Vec3&  position()  const { return position_;  }
constexpr const Color& intensity() const { return intensity_; }
};
inline std::ostream& operator<<(std::ostream& os, const ILight& light) {
os << light.description();
return os;
}




class PointLight final : public virtual ILight {
public:
PointLight(const Vec3& position, const Color& intensity);
PointLight(const Vec3& position, const Color& intensity, float constant, float linear, float quadratic);

virtual Color computeIntensityAtPoint(const Vec3& point) const override;
virtual std::string description() const override;

float attenuationConstant()  const;
float attenuationLinear()    const;
float attenuationQuadratic() const;

void setAttenuation(float constant, float linear, float quadratic);

private:
float attenuationConstant_;
float attenuationLinear_;
float attenuationQuadratic_;

static constexpr float DEFAULT_ATTENUATION_CONSTANT  = 1.00f;
static constexpr float DEFAULT_ATTENUATION_LINEAR    = 0.00f;
static constexpr float DEFAULT_ATTENUATION_QUADRATIC = 0.00f;
};
