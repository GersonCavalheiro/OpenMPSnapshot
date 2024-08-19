#pragma once
#include "bird.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
class Flock
{
public:
int number_of_birds = 100;
Bird **birds;
Flock();
~Flock();
void run();
void generate(Dimension);
};
