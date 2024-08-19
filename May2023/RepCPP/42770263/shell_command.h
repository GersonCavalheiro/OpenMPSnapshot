#pragma once
#include "typedefs.h"
#include <vector>
#include <string>
#include <iostream>

class ShellCommand { 
protected:
std::string name;
public:
ShellCommand();

virtual void run(std::string arguments);

std::string getName() {
return name;
}

bool checkNumberOfArguments(int real, int expected, std::ostream& out);

bool checkNumberOfArguments(int real, int expected1, int expected2, std::ostream& out);

std::vector<std::string> parseArguments(std::string notParsedArguments);

};
