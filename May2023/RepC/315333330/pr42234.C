extern int foo (void);
void
bar (int x)
{
#pragma omp critical
{
int j;
for (j = 0; j < foo (); j++)
;
if (0)
if (x >= 4)
;
}
}
