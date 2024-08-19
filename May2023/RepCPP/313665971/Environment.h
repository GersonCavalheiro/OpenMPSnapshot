#pragma once

#include <iostream>
#include <math.h>

enum Objects : int
{
Nothing = 0,
Line,
End
};

enum LineTypes : int
{
Horizontal = 0,
Vertical
};

typedef struct {
int x;
int y;
} Point;

class Environment
{
private:
int AreaWidth;
int AreaHeight;
int* content;

void clrEnv();
public:
Environment(int width, int height, int* env);
~Environment();

int getWidth();
int getHeight();
float getReward(int _x, int _y, int* _done);
void NakresliHernySvet(int _x, int _y);
int getState(int _x, int _y);
Objects getContent(int _x, int _y);
};


Environment::Environment(int width, int height, int* env)
{
this->AreaWidth = width;
this->AreaHeight = height;
this->content = env;
}

int Environment::getWidth()
{
return this->AreaWidth;
}

int Environment::getHeight()
{
return this->AreaHeight;
}

void Environment::clrEnv()
{
for (int i = 0; i < (this->AreaWidth * this->AreaHeight); i++)
this->content[i] = Objects::Nothing;
}

float Environment::getReward(int _x, int _y, int* _done)
{
float reward;

switch (this->content[getState(_x, _y)])
{
case Objects::Nothing:
reward = -0.55;
*_done = 0;
break;
case Objects::End:
reward = +1.0;
*_done = 1;
break;
default:
reward = -0.17;
*_done = 0;
break;
}

return reward;
}

void Environment::NakresliHernySvet(int _x, int _y)
{
system("clear");

for (int j = 0; j < AreaHeight; j++)
{
std::cout << "|";
for (int i = 0; i < AreaWidth; i++)
{
if (_x == i && _y == j)
{
std::cout << "\033[0;41m A \033[0m";
}
else
{
switch (this->content[(j * AreaWidth) + i])
{
case (int)Objects::Nothing:
std::cout << "\033[0m   ";
break;
case (int)Objects::Line:
std::cout << "\033[0;47;30m + \033[0m";
break;
case (int)Objects::End:
std::cout << "\033[0;42;30m E \033[0m";
break;
}
}
}
std::cout << "|" << std::endl;
}
std::cout << std::endl;
}

int Environment::getState(int _x, int _y)
{
return ((_y * this->AreaWidth) + _x);
}

Objects Environment::getContent(int _x, int _y)
{
return (Objects) this->content[getState(_x, _y)];
}

Environment::~Environment()
{

}
