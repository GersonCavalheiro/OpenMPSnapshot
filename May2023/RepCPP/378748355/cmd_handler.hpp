

#pragma once

#include "sdk.hpp"
#include <unordered_map>
#include "console_impl.hpp"

using namespace Impl;

using CommandHandlerFuncType = void (*)(const String& params, const ConsoleCommandSenderData& sender, ConsoleComponent& console, ICore* core);

class ConsoleCmdHandler
{
public:
static FlatHashMap<String, CommandHandlerFuncType> Commands;

ConsoleCmdHandler(const String& command, CommandHandlerFuncType handler)
{
auto it = Commands.find(command);
if (it != Commands.end())
{
it->second = handler;
}
else
{
Commands.insert({ command, handler });
}
command_ = command;
handler_ = handler;
}

~ConsoleCmdHandler()
{
Commands.erase(command_);
}

private:
String command_;
CommandHandlerFuncType handler_;
};

#define ADD_CONSOLE_CMD(cmd, handler) \
ConsoleCmdHandler console_command_##cmd(#cmd, handler)
