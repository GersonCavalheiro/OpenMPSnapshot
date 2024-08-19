void foo(int a)
{
int i;
#pragma analysis_check assert induction_var()
for (i=0; i<10; i++)
{
i += a;
}
}
void bar(int a)
{
int i;
#pragma analysis_check assert induction_var(i:0:9:a+a)
for (i=0; i<10; )
{
i += a + a;
}
}
