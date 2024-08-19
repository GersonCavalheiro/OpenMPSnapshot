#pragma once
#include "Math.hpp"
#include <iostream>


namespace detail {

inline static constexpr int   MIN_8BIT     = 0;
inline static constexpr int   MAX_8BIT     = 255;
inline static constexpr float INV_MAX_8BIT = 1.0f / MAX_8BIT;

inline static constexpr float byteToFloat(unsigned char v) {
return static_cast<int>(v) * INV_MAX_8BIT;
}

inline static constexpr unsigned char floatToByte(float v) {
return static_cast<unsigned char>(v * MAX_8BIT);
}

inline static constexpr unsigned char extractRedByte(unsigned int hex) {
return static_cast<unsigned char>((hex >> 16) & 0xff);
}

inline static constexpr unsigned char extractGreenByte(unsigned int hex) {
return static_cast<unsigned char>((hex >> 8) & 0xff);
}

inline static constexpr unsigned char extractBlueByte(unsigned int hex) {
return static_cast<unsigned char>(hex & 0xff);
}
}



class Color {
public:
float r;
float g;
float b;

Color() = default;
constexpr Color(const Color& color)           : r(color.r), g(color.g), b(color.b) {}
constexpr Color(const Color&& color) noexcept : r(color.r), g(color.g), b(color.b) {}

constexpr Color(float r, float g, float b) noexcept
: r(Math::clamp01(r)), g(Math::clamp01(g)), b(Math::clamp01(b)) {}

constexpr Color(double r, double g, double b) noexcept
: Color(static_cast<float>(r), static_cast<float>(g), static_cast<float>(b)) {}

explicit constexpr Color(unsigned int hex) noexcept
: Color(detail::extractRedByte(hex), detail::extractGreenByte(hex), detail::extractBlueByte(hex)) {}

constexpr Color(unsigned char r, unsigned char g, unsigned char b) noexcept
: Color(detail::byteToFloat(r), detail::byteToFloat(g), detail::byteToFloat(b)) {}


inline constexpr Color& operator=(const Color&& rhs) noexcept {
setClamped(rhs.r, rhs.g, rhs.b);
return *this;
}

inline constexpr Color& operator=(const Color& rhs) {
setClamped(rhs.r, rhs.g, rhs.b);
return *this;
}

inline constexpr Color& operator+=(const Color& rhs) {
setClamped(r + rhs.r, g + rhs.g, b + rhs.b);
return *this;
}

inline constexpr Color& operator-=(const Color& rhs) {
setClamped(r - rhs.r, g - rhs.g, b - rhs.b);
return *this;
}

inline constexpr Color& operator*=(const Color& rhs) {
setClamped(r * rhs.r, g * rhs.g, b * rhs.b);
return *this;
}

inline constexpr Color& operator*=(float rhs) {
setClamped(r * rhs, g * rhs, b * rhs);
return *this;
}

inline constexpr Color& operator/=(float rhs) {
setClamped(r / rhs, g / rhs, b / rhs);
return *this;
}

private:
inline constexpr void setClamped(float r, float g, float b) {
this->r = Math::clamp01(r);
this->g = Math::clamp01(g);
this->b = Math::clamp01(b);
}
};

inline constexpr Color operator+(const Color& lhs, const Color& rhs) {
return Color(lhs.r + rhs.r, lhs.g + rhs.g, lhs.b + rhs.b);
}

inline constexpr Color operator-(const Color& lhs, const Color& rhs) {
return Color(lhs.r - rhs.r, lhs.g - rhs.g, lhs.b - rhs.b);
}

inline constexpr Color operator*(const Color& lhs, const Color& rhs) {
return Color(lhs.r * rhs.r, lhs.g * rhs.g, lhs.b * rhs.b);
}

inline constexpr Color operator*(float lhs, const Color& rhs) {
return Color(lhs * rhs.r, lhs * rhs.g, lhs * rhs.b);
}

inline constexpr Color operator*(const Color& lhs, float rhs) {
return Color(lhs.r * rhs, lhs.g * rhs, lhs.b * rhs);
}

inline constexpr Color operator/(const Color& lhs, float rhs) {
return Color(lhs.r / rhs, lhs.g / rhs, lhs.b / rhs);
}

inline std::ostream& operator<<(std::ostream& os, const Color& color) {
os << color.r << "," << color.g << "," << color.b;
return os;
}




namespace Palette {
constexpr Color white   { 1.00f, 1.00f, 1.00f };
constexpr Color red     { 1.00f, 0.00f, 0.00f };
constexpr Color green   { 0.00f, 1.00f, 0.00f };
constexpr Color blue    { 0.00f, 0.00f, 1.00f };
constexpr Color cyan    { 0.00f, 1.00f, 1.00f };
constexpr Color magenta { 1.00f, 0.00f, 1.00f };
constexpr Color yellow  { 1.00f, 1.00f, 0.00f };
constexpr Color black   { 0.00f, 0.00f, 0.00f };

constexpr Color aliceBlue            { 0.941176f, 0.972549f, 1.000000f };
constexpr Color antiqueWhite         { 0.980392f, 0.921569f, 0.843137f };
constexpr Color aqua                 { 0.000000f, 0.000000f, 1.000000f };
constexpr Color aquamarine           { 0.498039f, 1.000000f, 0.831373f };
constexpr Color azure                { 0.941176f, 1.000000f, 1.000000f };
constexpr Color beige                { 0.960784f, 0.960784f, 0.862745f };
constexpr Color bisque               { 1.000000f, 0.894118f, 0.768627f };
constexpr Color blanchedAlmond       { 1.000000f, 0.921569f, 0.803922f };
constexpr Color blueViolet           { 0.541176f, 0.168627f, 0.886275f };
constexpr Color brown                { 0.647059f, 0.164706f, 0.164706f };
constexpr Color burlyWood            { 0.870588f, 0.721569f, 0.529412f };
constexpr Color cadetBlue            { 0.372549f, 0.619608f, 0.627451f };
constexpr Color chartreuse           { 0.498039f, 1.000000f, 0.000000f };
constexpr Color chocolate            { 0.823529f, 0.411765f, 0.117647f };
constexpr Color coral                { 1.000000f, 0.498039f, 0.313726f };
constexpr Color cornflowerBlue       { 0.392157f, 0.584314f, 0.929412f };
constexpr Color cornsilk             { 1.000000f, 0.972549f, 0.862745f };
constexpr Color crimson              { 0.862745f, 0.078431f, 0.235294f };
constexpr Color darkBlue             { 0.000000f, 0.000000f, 0.545098f };
constexpr Color darkCyan             { 0.000000f, 0.545098f, 0.545098f };
constexpr Color darkGray             { 0.662745f, 0.662745f, 0.662745f };
constexpr Color darkGreen            { 0.000000f, 0.392157f, 0.000000f };
constexpr Color darkKhaki            { 0.741176f, 0.717647f, 0.419608f };
constexpr Color darkMagenta          { 0.545098f, 0.000000f, 0.545098f };
constexpr Color darkOliveGreen       { 0.333333f, 0.419608f, 0.184314f };
constexpr Color darkorange           { 1.000000f, 0.549020f, 0.000000f };
constexpr Color darkOrchid           { 0.600000f, 0.196078f, 0.800000f };
constexpr Color darkRed              { 0.545098f, 0.000000f, 0.000000f };
constexpr Color darkSalmon           { 0.913725f, 0.588235f, 0.478431f };
constexpr Color darkSeaGreen         { 0.560784f, 0.737255f, 0.560784f };
constexpr Color darkSlateBlue        { 0.282353f, 0.239216f, 0.545098f };
constexpr Color darkSlateGray        { 0.184314f, 0.309804f, 0.309804f };
constexpr Color darkTurquoise        { 0.000000f, 0.807843f, 0.819608f };
constexpr Color darkViolet           { 0.580392f, 0.000000f, 0.827451f };
constexpr Color deepPink             { 1.000000f, 0.078431f, 0.576471f };
constexpr Color deepSkyBlue          { 0.000000f, 0.749020f, 1.000000f };
constexpr Color dimGray              { 0.411765f, 0.411765f, 0.411765f };
constexpr Color dodgerBlue           { 0.117647f, 0.564706f, 1.000000f };
constexpr Color fireBrick            { 0.698039f, 0.133333f, 0.133333f };
constexpr Color floralWhite          { 1.000000f, 0.980392f, 0.941176f };
constexpr Color forestGreen          { 0.133333f, 0.545098f, 0.133333f };
constexpr Color fuchsia              { 1.000000f, 0.000000f, 1.000000f };
constexpr Color gainsboro            { 0.862745f, 0.862745f, 0.862745f };
constexpr Color ghostWhite           { 0.972549f, 0.972549f, 1.000000f };
constexpr Color gold                 { 1.000000f, 0.843137f, 0.000000f };
constexpr Color goldenRod            { 0.854902f, 0.647059f, 0.125490f };
constexpr Color gray                 { 0.501961f, 0.501961f, 0.501961f };
constexpr Color greenYellow          { 0.678431f, 1.000000f, 0.184314f };
constexpr Color honeyDew             { 0.941176f, 1.000000f, 0.941176f };
constexpr Color hotPink              { 1.000000f, 0.411765f, 0.705882f };
constexpr Color indianRed            { 0.803922f, 0.360784f, 0.360784f };
constexpr Color indigo               { 0.294118f, 0.000000f, 0.509804f };
constexpr Color ivory                { 1.000000f, 1.000000f, 0.941176f };
constexpr Color khaki                { 0.941176f, 0.901961f, 0.549020f };
constexpr Color lavender             { 0.901961f, 0.901961f, 0.980392f };
constexpr Color lavenderBlush        { 1.000000f, 0.941176f, 0.960784f };
constexpr Color lawnGreen            { 0.486275f, 0.988235f, 0.000000f };
constexpr Color lemonChiffon         { 1.000000f, 0.980392f, 0.803922f };
constexpr Color lightBlue            { 0.678431f, 0.847059f, 0.901961f };
constexpr Color lightCoral           { 0.941176f, 0.501961f, 0.501961f };
constexpr Color lightCyan            { 0.878431f, 1.000000f, 1.000000f };
constexpr Color lightGoldenRodYellow { 0.980392f, 0.980392f, 0.823529f };
constexpr Color lightGray            { 0.827451f, 0.827451f, 0.827451f };
constexpr Color lightGreen           { 0.564706f, 0.933333f, 0.564706f };
constexpr Color lightPink            { 1.000000f, 0.713726f, 0.756863f };
constexpr Color lightSalmon          { 1.000000f, 0.627451f, 0.478431f };
constexpr Color lightSeaGreen        { 0.125490f, 0.698039f, 0.666667f };
constexpr Color lightSkyBlue         { 0.529412f, 0.807843f, 0.980392f };
constexpr Color lightSlateGray       { 0.466667f, 0.533333f, 0.600000f };
constexpr Color lightSteelBlue       { 0.690196f, 0.768627f, 0.870588f };
constexpr Color lightYellow          { 1.000000f, 1.000000f, 0.878431f };
constexpr Color lime                 { 0.000000f, 1.000000f, 0.000000f };
constexpr Color limeGreen            { 0.196078f, 0.803922f, 0.196078f };
constexpr Color linen                { 0.980392f, 0.941176f, 0.901961f };
constexpr Color maroon               { 0.501961f, 0.000000f, 0.000000f };
constexpr Color mediumAquaMarine     { 0.400000f, 0.803922f, 0.666667f };
constexpr Color mediumBlue           { 0.000000f, 0.000000f, 0.803922f };
constexpr Color mediumOrchid         { 0.729412f, 0.333333f, 0.827451f };
constexpr Color mediumPurple         { 0.576471f, 0.439216f, 0.858824f };
constexpr Color mediumSeaGreen       { 0.235294f, 0.701961f, 0.443137f };
constexpr Color mediumSlateBlue      { 0.482353f, 0.407843f, 0.933333f };
constexpr Color mediumSpringGreen    { 0.000000f, 0.980392f, 0.603922f };
constexpr Color mediumTurquoise      { 0.282353f, 0.819608f, 0.800000f };
constexpr Color mediumVioletRed      { 0.780392f, 0.082353f, 0.521569f };
constexpr Color midnightBlue         { 0.098039f, 0.098039f, 0.439216f };
constexpr Color mintCream            { 0.960784f, 1.000000f, 0.980392f };
constexpr Color mistyRose            { 1.000000f, 0.894118f, 0.882353f };
constexpr Color moccasin             { 1.000000f, 0.894118f, 0.709804f };
constexpr Color navajoWhite          { 1.000000f, 0.870588f, 0.678431f };
constexpr Color navy                 { 0.000000f, 0.000000f, 0.501961f };
constexpr Color oldLace              { 0.992157f, 0.960784f, 0.901961f };
constexpr Color olive                { 0.501961f, 0.501961f, 0.000000f };
constexpr Color oliveDrab            { 0.419608f, 0.556863f, 0.137255f };
constexpr Color orange               { 1.000000f, 0.647059f, 0.000000f };
constexpr Color orangeRed            { 1.000000f, 0.270588f, 0.000000f };
constexpr Color orchid               { 0.854902f, 0.439216f, 0.839216f };
constexpr Color paleGoldenRod        { 0.933333f, 0.909804f, 0.666667f };
constexpr Color paleGreen            { 0.596078f, 0.984314f, 0.596078f };
constexpr Color paleTurquoise        { 0.686275f, 0.933333f, 0.933333f };
constexpr Color paleVioletRed        { 0.858824f, 0.439216f, 0.576471f };
constexpr Color papayaWhip           { 1.000000f, 0.937255f, 0.835294f };
constexpr Color peachPuff            { 1.000000f, 0.854902f, 0.725490f };
constexpr Color peru                 { 0.803922f, 0.521569f, 0.247059f };
constexpr Color pink                 { 1.000000f, 0.752941f, 0.796078f };
constexpr Color plum                 { 0.866667f, 0.627451f, 0.866667f };
constexpr Color powderBlue           { 0.690196f, 0.878431f, 0.901961f };
constexpr Color purple               { 0.501961f, 0.000000f, 0.501961f };
constexpr Color rosyBrown            { 0.737255f, 0.560784f, 0.560784f };
constexpr Color royalBlue            { 0.254902f, 0.411765f, 0.882353f };
constexpr Color saddleBrown          { 0.545098f, 0.270588f, 0.074510f };
constexpr Color salmon               { 0.980392f, 0.501961f, 0.447059f };
constexpr Color sandyBrown           { 0.956863f, 0.643137f, 0.376471f };
constexpr Color seaGreen             { 0.180392f, 0.545098f, 0.341176f };
constexpr Color seaShell             { 1.000000f, 0.960784f, 0.933333f };
constexpr Color sienna               { 0.627451f, 0.321569f, 0.176471f };
constexpr Color silver               { 0.752941f, 0.752941f, 0.752941f };
constexpr Color skyBlue              { 0.529412f, 0.807843f, 0.921569f };
constexpr Color slateBlue            { 0.415686f, 0.352941f, 0.803922f };
constexpr Color slateGray            { 0.439216f, 0.501961f, 0.564706f };
constexpr Color snow                 { 1.000000f, 0.980392f, 0.980392f };
constexpr Color springGreen          { 0.000000f, 1.000000f, 0.498039f };
constexpr Color steelBlue            { 0.274510f, 0.509804f, 0.705882f };
constexpr Color blueSteel            { 0.274510f, 0.509804f, 0.705882f };
constexpr Color tan                  { 0.823529f, 0.705882f, 0.549020f };
constexpr Color teal                 { 0.000000f, 0.501961f, 0.501961f };
constexpr Color thistle              { 0.847059f, 0.749020f, 0.847059f };
constexpr Color tomato               { 1.000000f, 0.388235f, 0.278431f };
constexpr Color turquoise            { 0.250980f, 0.878431f, 0.815686f };
constexpr Color violet               { 0.933333f, 0.509804f, 0.933333f };
constexpr Color wheat                { 0.960784f, 0.870588f, 0.701961f };
constexpr Color whiteSmoke           { 0.960784f, 0.960784f, 0.960784f };
constexpr Color yellowGreen          { 0.603922f, 0.803922f, 0.196078f };
}
