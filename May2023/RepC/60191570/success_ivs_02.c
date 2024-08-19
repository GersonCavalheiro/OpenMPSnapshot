void f(bool b)
{
int i, max;
if (b)
{
i = 0;
max = 10;
}
else
{
i = 1;
max = 11;
}
#pragma analysis_check assert reaching_definition_in(i: 0,1; max: 10, 11) induction_var(i:0,1:-1+max:1)
for (; i<max; ++i)
{}
}
