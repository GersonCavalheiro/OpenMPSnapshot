#ifndef GENETICALGORITHM_RECTANGLE_H
#define GENETICALGORITHM_RECTANGLE_H
#include "Point.h"
#include "Color.h"
#pragma offload_attribute(push, target(mic))
struct Rectangle
{
Rectangle();
Rectangle(Point leftUp, Point rightDown,  Color color);
Point rightDown, leftUp;
Color color;
};
#pragma offload_attribute(pop)
#endif 
