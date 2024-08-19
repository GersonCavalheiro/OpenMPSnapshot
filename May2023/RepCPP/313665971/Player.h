#pragma once

#include <iostream>
#include <math.h>

enum Actions
{
Up = 0,
Down,
Left,
Right
};

class Player
{
private:
int positionX;
int positionY;

float reward;
public:
Player();
~Player();

void Move(int _action);
void Move(int _x, int _y);
int getX();
int getY();
void setReward(float val);
float getReward();
};

void Player::setReward(float val)
{
this->reward = val;
}

float Player::getReward()
{
return this->reward;
}

int Player::getX()
{
return this->positionX;
}

int Player::getY()
{
return this->positionY;
}

void Player::Move(int _action)
{
switch (_action)
{
case (int)Actions::Up:
this->positionY -= 1;
break;
case (int)Actions::Down:
this->positionY += 1;
break;
case (int)Actions::Left:
this->positionX -= 1;
break;
case (int)Actions::Right:
this->positionX += 1;
break;
}
}

void Player::Move(int _x, int _y)
{
this->positionX = _x;
this->positionY = _y;
}

Player::Player()
{
this->positionX = 2;
this->positionY = 2;

this->reward = 0;
}


Player::~Player()
{
}
