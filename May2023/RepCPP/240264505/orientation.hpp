

#ifndef LBT_BOUNDARY_ORIENTATION
#define LBT_BOUNDARY_ORIENTATION
#pragma once

#include <cstdint>
#include <ostream>


namespace lbt {
namespace boundary{


enum class Orientation { Left, Right, Front, Back, Bottom, Top };


inline constexpr Orientation operator! (Orientation const& orientation) noexcept {
switch (orientation) {
case Orientation::Left:
return Orientation::Right;
case Orientation::Right:
return Orientation::Left;
case Orientation::Front:
return Orientation::Back;
case Orientation::Back:
return Orientation::Front;
case Orientation::Bottom:
return Orientation::Top;
case Orientation::Top:
return Orientation::Bottom;
default:
return orientation;
}
}


inline std::ostream& operator<< (std::ostream& os, Orientation const orientation) noexcept {
switch (orientation) {
case Orientation::Left:
os << "left";
break;
case Orientation::Right:
os << "right";
break;
case Orientation::Front:
os << "front";
break;
case Orientation::Back:
os << "back";
break;
case Orientation::Bottom:
os << "bottom";
break;
case Orientation::Top:
os << "top";
break;
default:
os << "none";
break;
}
return os;
}

}
}

#endif 
