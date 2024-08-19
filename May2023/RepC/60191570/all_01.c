void f(int x)
{
int k;
#pragma analysis_check assert defined(k)
if (x > 3)
k = 1;
else
k = 2;
#pragma analysis_check assert reaching_definition_in(k: 1, 2)
if (x > 3) {
#pragma analysis_check assert live_in(x, k)
k = k + x;
}
int i = 0;
#pragma analysis_check assert dead(k)
k = 0;
#pragma analysis_check assert reaching_definition_in(i: 0) induction_var(i:0:99:1)
while (i < 100) {
++i;
}
}
