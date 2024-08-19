int x = 0;
#pragma analysis_check assert_decl live_in(x) defined(x) upper_exposed(x)
int main(int argc, char *argv[])
{
int a, b;
#pragma analysis_check assert defined(a, b)
a = b = 0;
x++;
#pragma analysis_check assert_decl live_out(x)
if(x)
return 0;
#pragma analysis_check assert live_in(x) dead(b)
b = x + 20;
return 0;
}
