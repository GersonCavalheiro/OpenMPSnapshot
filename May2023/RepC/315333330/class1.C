#pragma GCC visibility push(hidden)
struct __attribute__ ((visibility ("default"))) nsINIParser
{
static void Init();
};
__attribute__ ((visibility ("default")))
void
CheckCompatibility(void)
{
nsINIParser::Init();
}
