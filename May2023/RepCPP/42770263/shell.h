#pragma once
#include "shell_command.h"
#include <map>
#define PROMPT '$'

class Shell {
private:
std::map <std::string, ShellCommand* > commands;
std::ostream& out;
std::istream& in;
public:
Shell(const std::map<std::string, ShellCommand*>& newCommands, std::ostream& out, std::istream& in);
void start ();
private:
std::string getCommandName(const std::string& commandWithArgs) const;
bool isSpaces(const std::string& str) const;
};
