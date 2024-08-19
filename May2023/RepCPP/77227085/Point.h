

#pragma once

#include "rawspeedconfig.h" 
#include <algorithm>        
#include <cmath>            
#include <cstdint>          
#include <cstdlib>          
#include <tuple>            
#include <type_traits>      

namespace rawspeed {

class iPoint2D {
public:
using value_type = int32_t;
using area_type = uint64_t;

constexpr iPoint2D() = default;
constexpr iPoint2D(value_type a, value_type b) : x(a), y(b) {}

constexpr iPoint2D operator+(const iPoint2D& rhs) const {
return {x + rhs.x, y + rhs.y};
}
constexpr iPoint2D operator-(const iPoint2D& rhs) const {
return {x - rhs.x, y - rhs.y};
}

iPoint2D& operator+=(const iPoint2D& rhs) {
*this = operator+(rhs);
return *this;
}
iPoint2D& operator-=(const iPoint2D& rhs) {
*this = operator-(rhs);
return *this;
}

constexpr bool operator==(const iPoint2D& rhs) const {
return x == rhs.x && y == rhs.y;
}
constexpr bool operator!=(const iPoint2D& rhs) const {
return !operator==(rhs);
}

constexpr bool operator>(const iPoint2D& rhs) const {
return x > rhs.x && y > rhs.y;
}
constexpr bool operator<(const iPoint2D& rhs) const {
return x < rhs.x && y < rhs.y;
}

constexpr bool operator>=(const iPoint2D& rhs) const {
return x >= rhs.x && y >= rhs.y;
}
constexpr bool operator<=(const iPoint2D& rhs) const {
return x <= rhs.x && y <= rhs.y;
}

[[nodiscard]] bool hasPositiveArea() const { return operator>({0, 0}); }

[[nodiscard]] area_type RAWSPEED_READONLY area() const {
using signed_area = std::make_signed_t<area_type>;

area_type x_abs = std::abs(static_cast<signed_area>(x));
area_type y_abs = std::abs(static_cast<signed_area>(y));

return x_abs * y_abs;
}

[[nodiscard]] constexpr bool isThisInside(const iPoint2D& rhs) const {
return operator<=(rhs);
}

[[nodiscard]] constexpr iPoint2D getSmallest(const iPoint2D& rhs) const {
return {
std::min(x, rhs.x),
std::min(y, rhs.y),
};
}

value_type x = 0;
value_type y = 0;
};


class iRectangle2D {
public:
constexpr iRectangle2D() = default;
constexpr iRectangle2D(const iPoint2D& pos_, const iPoint2D& dim_)
: pos(pos_), dim(dim_) {}

constexpr iRectangle2D(int w, int h) : dim({w, h}) {}
constexpr iRectangle2D(int x_pos, int y_pos, int w, int h)
: pos({x_pos, y_pos}), dim({w, h}) {}

[[nodiscard]] constexpr int getTop() const { return pos.y; }
[[nodiscard]] constexpr int getBottom() const { return pos.y + dim.y; }
[[nodiscard]] constexpr int getLeft() const { return pos.x; }
[[nodiscard]] constexpr int getRight() const { return pos.x + dim.x; }
[[nodiscard]] constexpr int getWidth() const { return dim.x; }
[[nodiscard]] constexpr int getHeight() const { return dim.y; }
[[nodiscard]] constexpr iPoint2D getTopLeft() const { return pos; }
[[nodiscard]] constexpr iPoint2D getBottomRight() const { return pos + dim; }
[[nodiscard]] constexpr iPoint2D getTopRight() const {
return pos + iPoint2D(dim.x, 0);
}
[[nodiscard]] constexpr iPoint2D getBottomLeft() const {
return pos + iPoint2D(0, dim.y);
}
[[nodiscard]] constexpr bool hasPositiveArea() const {
return (dim.x > 0) && (dim.y > 0);
}

[[nodiscard]] constexpr bool isPointInside(const iPoint2D& subPoint) const {
return subPoint >= getTopLeft() && subPoint < getBottomRight();
}

[[nodiscard]] constexpr bool
isPointInsideInclusive(const iPoint2D& subPoint) const {
return subPoint >= getTopLeft() && subPoint <= getBottomRight();
}

[[nodiscard]] constexpr bool
isThisInside(const iRectangle2D& superRect) const {
return getTopLeft() >= superRect.getTopLeft() &&
getBottomRight() <= superRect.getBottomRight();
}

[[nodiscard]] unsigned int area() const { return dim.area(); }

void offset(const iPoint2D& offset_) { pos += offset_; }


void setTopLeft(const iPoint2D& top_left) { pos = top_left; }


void setBottomRightAbsolute(const iPoint2D& bottom_right) {
dim = bottom_right - pos;
}

void setAbsolute(const iPoint2D& top_left, const iPoint2D& bottom_right) {
pos = top_left;
setBottomRightAbsolute(bottom_right);
}
void setAbsolute(int x1, int y1, int x2, int y2) {
setAbsolute({x1, y1}, {x2, y2});
}

void setSize(const iPoint2D& size) { dim = size; }



bool cropArea() {
dim.x = std::max(0, dim.x);
dim.y = std::max(0, dim.y);
return hasPositiveArea();
}



bool cropOffsetToZero() {
iPoint2D crop_pixels;
if (pos.x < 0) {
crop_pixels.x = -(pos.x);
pos.x = 0;
}
if (pos.y < 0) {
crop_pixels.y = -pos.y;
pos.y = 0;
}
dim -= crop_pixels;
return cropArea();
}

[[nodiscard]] iRectangle2D getOverlap(const iRectangle2D& other) const {
iRectangle2D overlap;
iPoint2D br1 = getBottomRight();
iPoint2D br2 = other.getBottomRight();
overlap.setAbsolute(std::max(pos.x, other.pos.x),
std::max(pos.y, other.pos.y), std::min(br1.x, br2.x),
std::min(br1.y, br2.y));
return overlap;
}

[[nodiscard]] iRectangle2D combine(const iRectangle2D& other) const {
iRectangle2D combined;
iPoint2D br1 = getBottomRight();
iPoint2D br2 = other.getBottomRight();
combined.setAbsolute(std::min(pos.x, other.pos.x),
std::min(pos.y, other.pos.y), std::max(br1.x, br2.x),
std::max(br2.y, br2.y));
return combined;
}

iPoint2D pos{0, 0};
iPoint2D dim{0, 0};
};

inline bool operator==(const iRectangle2D& a, const iRectangle2D b) {
return std::tie(a.pos, a.dim) == std::tie(b.pos, b.dim);
}
inline bool operator!=(const iRectangle2D& a, const iRectangle2D b) {
return !(a == b);
}

} 
