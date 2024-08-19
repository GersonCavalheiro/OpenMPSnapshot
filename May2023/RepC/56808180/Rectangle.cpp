#pragma once
#include "Rectangle.h"
Rectangle::Rectangle(Point leftUp, Point rightDown, Color color) :
leftUp(leftUp), rightDown(rightDown), color(color) {
}
Rectangle::Rectangle() { };