#include <string>
std::string UNKNOWN = "UNKNOWN";
int foo(int n)
{
int result;
if (n > 2)
{
if (n < 100)
{
if (n > 50)
return 1;
return 2;
}
else
{
#pragma analysis_check assert live_in(n) live_out(n)
result = n / 0;
}
#pragma analysis_check assert reaching_definition_in(n: UNKNOWN) reaching_definition_out(result: n/0)
result = n / 0;
}
#pragma analysis_check assert reaching_definition_in(result: n/0)
return result;
}