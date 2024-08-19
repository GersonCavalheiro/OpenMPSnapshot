#include <string>
std::string UNKNOWN = "UNKNOWN";
std::string UNDEFINED = "UNDEFINED";
bool global_var = true;
void f(int x)
{
int k;
#pragma analysis_check assert reaching_definition_in(k: UNDEFINED; global_var: UNDEFINED; x: UNKNOWN) reaching_definition_out(k: 5; global_var: false; x: 10)
{
k=5;
global_var = false;
x=10;
}
}