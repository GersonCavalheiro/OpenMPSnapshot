
#pragma once

#include <string>
#include <vector>


#include "includes/define.h"

namespace Kratos
{





namespace StringUtilities
{

std::string KRATOS_API(KRATOS_CORE) ConvertCamelCaseToSnakeCase(const std::string& rString);


std::string KRATOS_API(KRATOS_CORE) ConvertSnakeCaseToCamelCase(const std::string& rString);


std::string KRATOS_API(KRATOS_CORE) ErasePartialString(
const std::string& rMainString,
const std::string& rToErase
);


bool KRATOS_API(KRATOS_CORE) ContainsPartialString(
const std::string& rMainString,
const std::string& rToCheck
);


std::string KRATOS_API(KRATOS_CORE) RemoveWhiteSpaces(const std::string& rString);


std::vector<std::string> KRATOS_API(KRATOS_CORE) SplitStringByDelimiter(
const std::string& rString,
const char Delimiter
);


std::string KRATOS_API(KRATOS_CORE) ReplaceAllSubstrings(
const std::string& rInputString,
const std::string& rStringToBeReplaced,
const std::string& rStringToReplace
);


template<class TClass>
static void PrintDataWithIdentation(
std::ostream& rOStream,
const TClass& rThisClass,
const std::string Identation = "\t"
)
{
std::stringstream ss;
std::string line;
rThisClass.PrintData(ss);

const std::string& r_output = ss.str();

std::istringstream iss(r_output);
while (std::getline(iss, line)) {
rOStream << Identation << line << "\n";
}
}

}; 
}  