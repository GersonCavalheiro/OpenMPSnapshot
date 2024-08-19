#pragma once
#include <math.h>
class Vector
{
public:
Vector();
Vector(float, float);
~Vector();
float length();
float x = 0;
float y = 0;
void normalize(float);
void subtract(Vector);
void add(Vector);
void divide(float);
void multiply(float);
void limit(float);
static float distance(Vector, Vector);
static Vector subtract(Vector, Vector);
static Vector add(Vector, Vector);
};
