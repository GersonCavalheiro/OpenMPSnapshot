
#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "StringTools.hpp"

namespace CLI {

#define CLI11_ERROR_DEF(parent, name)                                                                                  \
protected:                                                                                                           \
name(std::string ename, std::string msg, int exit_code) : parent(std::move(ename), std::move(msg), exit_code) {}   \
name(std::string ename, std::string msg, ExitCodes exit_code)                                                      \
: parent(std::move(ename), std::move(msg), exit_code) {}                                                       \
\
public:                                                                                                              \
name(std::string msg, ExitCodes exit_code) : parent(#name, std::move(msg), exit_code) {}                           \
name(std::string msg, int exit_code) : parent(#name, std::move(msg), exit_code) {}

#define CLI11_ERROR_SIMPLE(name)                                                                                       \
explicit name(std::string msg) : name(#name, msg, ExitCodes::name) {}

enum class ExitCodes {
Success = 0,
IncorrectConstruction = 100,
BadNameString,
OptionAlreadyAdded,
FileError,
ConversionError,
ValidationError,
RequiredError,
RequiresError,
ExcludesError,
ExtrasError,
ConfigError,
InvalidError,
HorribleError,
OptionNotFound,
ArgumentMismatch,
BaseClass = 127
};



class Error : public std::runtime_error {
int actual_exit_code;
std::string error_name{"Error"};

public:
int get_exit_code() const { return actual_exit_code; }

std::string get_name() const { return error_name; }

Error(std::string name, std::string msg, int exit_code = static_cast<int>(ExitCodes::BaseClass))
: runtime_error(msg), actual_exit_code(exit_code), error_name(std::move(name)) {}

Error(std::string name, std::string msg, ExitCodes exit_code) : Error(name, msg, static_cast<int>(exit_code)) {}
};


class ConstructionError : public Error {
CLI11_ERROR_DEF(Error, ConstructionError)
};

class IncorrectConstruction : public ConstructionError {
CLI11_ERROR_DEF(ConstructionError, IncorrectConstruction)
CLI11_ERROR_SIMPLE(IncorrectConstruction)
static IncorrectConstruction PositionalFlag(std::string name) {
return IncorrectConstruction(name + ": Flags cannot be positional");
}
static IncorrectConstruction Set0Opt(std::string name) {
return IncorrectConstruction(name + ": Cannot set 0 expected, use a flag instead");
}
static IncorrectConstruction SetFlag(std::string name) {
return IncorrectConstruction(name + ": Cannot set an expected number for flags");
}
static IncorrectConstruction ChangeNotVector(std::string name) {
return IncorrectConstruction(name + ": You can only change the expected arguments for vectors");
}
static IncorrectConstruction AfterMultiOpt(std::string name) {
return IncorrectConstruction(
name + ": You can't change expected arguments after you've changed the multi option policy!");
}
static IncorrectConstruction MissingOption(std::string name) {
return IncorrectConstruction("Option " + name + " is not defined");
}
static IncorrectConstruction MultiOptionPolicy(std::string name) {
return IncorrectConstruction(name + ": multi_option_policy only works for flags and exact value options");
}
};

class BadNameString : public ConstructionError {
CLI11_ERROR_DEF(ConstructionError, BadNameString)
CLI11_ERROR_SIMPLE(BadNameString)
static BadNameString OneCharName(std::string name) { return BadNameString("Invalid one char name: " + name); }
static BadNameString BadLongName(std::string name) { return BadNameString("Bad long name: " + name); }
static BadNameString DashesOnly(std::string name) {
return BadNameString("Must have a name, not just dashes: " + name);
}
static BadNameString MultiPositionalNames(std::string name) {
return BadNameString("Only one positional name allowed, remove: " + name);
}
};

class OptionAlreadyAdded : public ConstructionError {
CLI11_ERROR_DEF(ConstructionError, OptionAlreadyAdded)
explicit OptionAlreadyAdded(std::string name)
: OptionAlreadyAdded(name + " is already added", ExitCodes::OptionAlreadyAdded) {}
static OptionAlreadyAdded Requires(std::string name, std::string other) {
return OptionAlreadyAdded(name + " requires " + other, ExitCodes::OptionAlreadyAdded);
}
static OptionAlreadyAdded Excludes(std::string name, std::string other) {
return OptionAlreadyAdded(name + " excludes " + other, ExitCodes::OptionAlreadyAdded);
}
};


class ParseError : public Error {
CLI11_ERROR_DEF(Error, ParseError)
};


class Success : public ParseError {
CLI11_ERROR_DEF(ParseError, Success)
Success() : Success("Successfully completed, should be caught and quit", ExitCodes::Success) {}
};

class CallForHelp : public Success {
CLI11_ERROR_DEF(Success, CallForHelp)
CallForHelp() : CallForHelp("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

class CallForAllHelp : public Success {
CLI11_ERROR_DEF(Success, CallForAllHelp)
CallForAllHelp()
: CallForAllHelp("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

class CallForVersion : public Success {
CLI11_ERROR_DEF(Success, CallForVersion)
CallForVersion()
: CallForVersion("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

class RuntimeError : public ParseError {
CLI11_ERROR_DEF(ParseError, RuntimeError)
explicit RuntimeError(int exit_code = 1) : RuntimeError("Runtime error", exit_code) {}
};

class FileError : public ParseError {
CLI11_ERROR_DEF(ParseError, FileError)
CLI11_ERROR_SIMPLE(FileError)
static FileError Missing(std::string name) { return FileError(name + " was not readable (missing?)"); }
};

class ConversionError : public ParseError {
CLI11_ERROR_DEF(ParseError, ConversionError)
CLI11_ERROR_SIMPLE(ConversionError)
ConversionError(std::string member, std::string name)
: ConversionError("The value " + member + " is not an allowed value for " + name) {}
ConversionError(std::string name, std::vector<std::string> results)
: ConversionError("Could not convert: " + name + " = " + detail::join(results)) {}
static ConversionError TooManyInputsFlag(std::string name) {
return ConversionError(name + ": too many inputs for a flag");
}
static ConversionError TrueFalse(std::string name) {
return ConversionError(name + ": Should be true/false or a number");
}
};

class ValidationError : public ParseError {
CLI11_ERROR_DEF(ParseError, ValidationError)
CLI11_ERROR_SIMPLE(ValidationError)
explicit ValidationError(std::string name, std::string msg) : ValidationError(name + ": " + msg) {}
};

class RequiredError : public ParseError {
CLI11_ERROR_DEF(ParseError, RequiredError)
explicit RequiredError(std::string name) : RequiredError(name + " is required", ExitCodes::RequiredError) {}
static RequiredError Subcommand(std::size_t min_subcom) {
if(min_subcom == 1) {
return RequiredError("A subcommand");
}
return RequiredError("Requires at least " + std::to_string(min_subcom) + " subcommands",
ExitCodes::RequiredError);
}
static RequiredError
Option(std::size_t min_option, std::size_t max_option, std::size_t used, const std::string &option_list) {
if((min_option == 1) && (max_option == 1) && (used == 0))
return RequiredError("Exactly 1 option from [" + option_list + "]");
if((min_option == 1) && (max_option == 1) && (used > 1)) {
return RequiredError("Exactly 1 option from [" + option_list + "] is required and " + std::to_string(used) +
" were given",
ExitCodes::RequiredError);
}
if((min_option == 1) && (used == 0))
return RequiredError("At least 1 option from [" + option_list + "]");
if(used < min_option) {
return RequiredError("Requires at least " + std::to_string(min_option) + " options used and only " +
std::to_string(used) + "were given from [" + option_list + "]",
ExitCodes::RequiredError);
}
if(max_option == 1)
return RequiredError("Requires at most 1 options be given from [" + option_list + "]",
ExitCodes::RequiredError);

return RequiredError("Requires at most " + std::to_string(max_option) + " options be used and " +
std::to_string(used) + "were given from [" + option_list + "]",
ExitCodes::RequiredError);
}
};

class ArgumentMismatch : public ParseError {
CLI11_ERROR_DEF(ParseError, ArgumentMismatch)
CLI11_ERROR_SIMPLE(ArgumentMismatch)
ArgumentMismatch(std::string name, int expected, std::size_t received)
: ArgumentMismatch(expected > 0 ? ("Expected exactly " + std::to_string(expected) + " arguments to " + name +
", got " + std::to_string(received))
: ("Expected at least " + std::to_string(-expected) + " arguments to " + name +
", got " + std::to_string(received)),
ExitCodes::ArgumentMismatch) {}

static ArgumentMismatch AtLeast(std::string name, int num, std::size_t received) {
return ArgumentMismatch(name + ": At least " + std::to_string(num) + " required but received " +
std::to_string(received));
}
static ArgumentMismatch AtMost(std::string name, int num, std::size_t received) {
return ArgumentMismatch(name + ": At Most " + std::to_string(num) + " required but received " +
std::to_string(received));
}
static ArgumentMismatch TypedAtLeast(std::string name, int num, std::string type) {
return ArgumentMismatch(name + ": " + std::to_string(num) + " required " + type + " missing");
}
static ArgumentMismatch FlagOverride(std::string name) {
return ArgumentMismatch(name + " was given a disallowed flag override");
}
};

class RequiresError : public ParseError {
CLI11_ERROR_DEF(ParseError, RequiresError)
RequiresError(std::string curname, std::string subname)
: RequiresError(curname + " requires " + subname, ExitCodes::RequiresError) {}
};

class ExcludesError : public ParseError {
CLI11_ERROR_DEF(ParseError, ExcludesError)
ExcludesError(std::string curname, std::string subname)
: ExcludesError(curname + " excludes " + subname, ExitCodes::ExcludesError) {}
};

class ExtrasError : public ParseError {
CLI11_ERROR_DEF(ParseError, ExtrasError)
explicit ExtrasError(std::vector<std::string> args)
: ExtrasError((args.size() > 1 ? "The following arguments were not expected: "
: "The following argument was not expected: ") +
detail::rjoin(args, " "),
ExitCodes::ExtrasError) {}
ExtrasError(const std::string &name, std::vector<std::string> args)
: ExtrasError(name,
(args.size() > 1 ? "The following arguments were not expected: "
: "The following argument was not expected: ") +
detail::rjoin(args, " "),
ExitCodes::ExtrasError) {}
};

class ConfigError : public ParseError {
CLI11_ERROR_DEF(ParseError, ConfigError)
CLI11_ERROR_SIMPLE(ConfigError)
static ConfigError Extras(std::string item) { return ConfigError("INI was not able to parse " + item); }
static ConfigError NotConfigurable(std::string item) {
return ConfigError(item + ": This option is not allowed in a configuration file");
}
};

class InvalidError : public ParseError {
CLI11_ERROR_DEF(ParseError, InvalidError)
explicit InvalidError(std::string name)
: InvalidError(name + ": Too many positional arguments with unlimited expected args", ExitCodes::InvalidError) {
}
};

class HorribleError : public ParseError {
CLI11_ERROR_DEF(ParseError, HorribleError)
CLI11_ERROR_SIMPLE(HorribleError)
};


class OptionNotFound : public Error {
CLI11_ERROR_DEF(Error, OptionNotFound)
explicit OptionNotFound(std::string name) : OptionNotFound(name + " not found", ExitCodes::OptionNotFound) {}
};

#undef CLI11_ERROR_DEF
#undef CLI11_ERROR_SIMPLE


}  
