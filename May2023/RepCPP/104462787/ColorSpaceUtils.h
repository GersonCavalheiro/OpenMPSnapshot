#ifndef CAPTURE3_COLOR_SPACE_UTILS_H
#define CAPTURE3_COLOR_SPACE_UTILS_H


#include <cmath>
#include <vector>
#include <omp.h>


namespace Capture3
{

static void RGBtoXYZ(const double *rgb, double *xyz, const unsigned int area, const bool gammaCorrected = false)
{

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;

double r = rgb[index + 0];
double g = rgb[index + 1];
double b = rgb[index + 2];

if (gammaCorrected) {
r = r > 0.04045 ? std::pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
g = g > 0.04045 ? std::pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
b = b > 0.04045 ? std::pow((b + 0.055) / 1.055, 2.4) : b / 12.92;
}

const double X = (r * 0.4360747) + (g * 0.3850649) + (b * 0.1430804);
const double Y = (r * 0.2225045) + (g * 0.7168786) + (b * 0.0606169);
const double Z = (r * 0.0139322) + (g * 0.0971045) + (b * 0.7141733);

xyz[index + 0] = X;
xyz[index + 1] = Y;
xyz[index + 2] = Z;
}
}


static void XYZtoRGB(const double *xyz, double *rgb, const unsigned int area, const bool gammaCorrected = false)
{

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;

const double X = xyz[index + 0];
const double Y = xyz[index + 1];
const double Z = xyz[index + 2];

double r = X * 3.1338561 + Y * -1.6168667 + Z * -0.4906146;
double g = X * -0.9787684 + Y * 1.9161415 + Z * 0.0334540;
double b = X * 0.0719453 + Y * -0.2289914 + Z * 1.4052427;

r = r < 0 ? 0 : r > 1 ? 1 : r;
g = g < 0 ? 0 : g > 1 ? 1 : g;
b = b < 0 ? 0 : b > 1 ? 1 : b;

if (gammaCorrected) {
r = r > 0.0031308 ? 1.055 * std::pow(r, 1.0 / 2.4) - 0.055 : 12.92 * r;
g = g > 0.0031308 ? 1.055 * std::pow(g, 1.0 / 2.4) - 0.055 : 12.92 * g;
b = b > 0.0031308 ? 1.055 * std::pow(b, 1.0 / 2.4) - 0.055 : 12.92 * b;
}

rgb[index + 0] = r;
rgb[index + 1] = g;
rgb[index + 2] = b;
}
}


static void LABtoXYZ(const double *lab, double *xyz, const unsigned int area)
{

const double K = 24389.0 / 27.0;
const double E = 216.0 / 24389.0;
const double C = 7.787;

const double D50X = 0.96422;
const double D50Y = 1.0;
const double D50Z = 0.82521;

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;

const double L = (lab[index + 0] * 100.0);
const double A = (lab[index + 1] * 256.0) - 128.0;
const double B = (lab[index + 2] * 256.0) - 128.0;

const double y = (L + 16.0) / 116.0;
const double x = A / 500.0 + y;
const double z = y - B / 200.0;

const double x3 = x * x * x;
const double z3 = z * z * z;

const double X = D50X * (x3 > E ? x3 : (x - 16.0 / 116.0) / C);
const double Y = D50Y * (L > (K * E) ? std::pow(((L + 16.0) / 116.0), 3) : L / K);
const double Z = D50Z * (z3 > E ? z3 : (z - 16.0 / 116.0) / C);

xyz[index + 0] = X;
xyz[index + 1] = Y;
xyz[index + 2] = Z;
}
}


static void XYZtoLAB(const double *xyz, double *lab, const unsigned int area)
{

const double K = 24389.0 / 27.0;
const double E = 216.0 / 24389.0;
const double C = 7.787;

const double D50X = 0.96422;
const double D50Y = 1.0;
const double D50Z = 0.82521;

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;

double X = xyz[index + 0];
double Y = xyz[index + 1];
double Z = xyz[index + 2];

X = X / D50X;
Y = Y / D50Y;
Z = Z / D50Z;

const double x = X > E ? std::pow(X, 1.0 / 3.0) : (C * X) + (16.0 / 116.0);
const double y = Y > E ? std::pow(Y, 1.0 / 3.0) : (C * Y) + (16.0 / 116.0);
const double z = Z > E ? std::pow(Z, 1.0 / 3.0) : (C * Z) + (16.0 / 116.0);

const double L = (116.0 * y) - 16.0;
const double A = 500.0 * (x - y);
const double B = 200.0 * (y - z);

lab[index + 0] = (L / 100.0);
lab[index + 1] = (A + 128.0) / 256.0;
lab[index + 2] = (B + 128.0) / 256.0;
}
}


static void XYVtoXYZ(const double *xyv, double *xyz, const unsigned int size)
{

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < size; i++) {

const unsigned int index = i * 3;

const double x = xyv[index + 0];
const double y = xyv[index + 1];
const double v = xyv[index + 2];

double X = x * (v / y);
double Y = v;
double Z = (1.0 - x - y) * (v / y);

if (y == 0) {
X = 0;
Y = 0;
Z = 0;
}

xyz[index + 0] = X;
xyz[index + 1] = Y;
xyz[index + 2] = Z;
}
}


static void XYZtoXYV(const double *xyz, double *xyv, const unsigned int area)
{

const double D50X = 0.96422;
const double D50Y = 1.0;
const double D50Z = 0.82521;

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;

const double X = xyz[index + 0];
const double Y = xyz[index + 1];
const double Z = xyz[index + 2];

double x = X / (X + Y + Z);
double y = Y / (X + Y + Z);
double v = Y;

if (X == 0 && Y == 0 && Z == 0) {
x = D50X / (D50X + D50Y + D50Z);
y = D50Y / (D50X + D50Y + D50Z);
v = 0;
}

xyv[index + 0] = x;
xyv[index + 1] = y;
xyv[index + 2] = v;
}
}


static void RGBtoHSB(const double *rgb, double *hsb, const unsigned int area, const bool gammaCorrected = false)
{
#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;

double r = rgb[index + 0];
double g = rgb[index + 1];
double b = rgb[index + 2];

if (!gammaCorrected) {
r = r > 0.0031308 ? 1.055 * std::pow(r, 1.0 / 2.4) - 0.055 : 12.92 * r;
g = g > 0.0031308 ? 1.055 * std::pow(g, 1.0 / 2.4) - 0.055 : 12.92 * g;
b = b > 0.0031308 ? 1.055 * std::pow(b, 1.0 / 2.4) - 0.055 : 12.92 * b;
}

const double min = std::min(std::min(r, g), b);
const double max = std::max(std::max(r, g), b);
const double delta = max - min;

double H = 0;
double S = 0;
double B = max;

if (delta > 0) {

if (max > 0) {
S = delta / max;
}

const double deltaR = (max - r) / delta;
const double deltaG = (max - g) / delta;
const double deltaB = (max - b) / delta;

if (r == max) {
H = deltaB - deltaG;
} else if (g == max) {
H = 2.0 + deltaR - deltaB;
} else {
H = 4.0 + deltaG - deltaR;
}

H /= 6.0;

if (H < 0) {
H += 1;
}
if (H >= 1) {
H -= 1;
}
}

H = H < 0 ? 0 : H > 1 ? 1 : H;
S = S < 0 ? 0 : S > 1 ? 1 : S;
B = B < 0 ? 0 : B > 1 ? 1 : B;

hsb[index + 0] = H;
hsb[index + 1] = S;
hsb[index + 2] = B;
}
}


static void HSBtoRGB(const double *hsb, double *rgb, const unsigned int area, const bool gammaCorrected = false)
{
#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;

double H = hsb[index + 0];
double S = hsb[index + 1];
double B = hsb[index + 2];

double r = 0;
double g = 0;
double b = 0;

if (S == 0) {
r = B;
g = B;
b = B;
} else {
H = H - std::floor(H);
const auto n = (int) (6.0 * H);
const double f = 6.0 * H - n;
const double p = B * (1.0 - S);
const double q = B * (1.0 - S * f);
const double t = B * (1.0 - (S * (1.0 - f)));
switch (n) {
case 0:
r = B;
g = t;
b = p;
break;
case 1:
r = q;
g = B;
b = p;
break;
case 2:
r = p;
g = B;
b = t;
break;
case 3:
r = p;
g = q;
b = B;
break;
case 4:
r = t;
g = p;
b = B;
break;
default:
r = B;
g = p;
b = q;
break;
}
}

if (gammaCorrected) {
r = r > 0.04045 ? std::pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
g = g > 0.04045 ? std::pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
b = b > 0.04045 ? std::pow((b + 0.055) / 1.055, 2.4) : b / 12.92;
}

r = r < 0 ? 0 : r > 1 ? 1 : r;
g = g < 0 ? 0 : g > 1 ? 1 : g;
b = b < 0 ? 0 : b > 1 ? 1 : b;

rgb[index + 0] = r;
rgb[index + 1] = g;
rgb[index + 2] = b;
}
}

}


#endif 
