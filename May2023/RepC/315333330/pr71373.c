void foo (void)
{
#pragma acc parallel
{
#pragma acc loop independent
for (int i = 0; i < 16; i++)
;
}
void bar (void)
{
}
}
void BAR (void)
{
void FOO (void)
{
#pragma acc parallel
{
#pragma acc loop independent
for (int i = 0; i < 16; i++)
;
}
}
}
