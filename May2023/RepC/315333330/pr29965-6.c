extern void baz () __attribute__ ((noreturn));
void
foo1 ()
{
int i;
#pragma omp for schedule (static, 16)
for (i = 0; i < 2834; i++)
for (;;)
;
}
void
bar1 ()
{
int i;
#pragma omp for schedule (static, 16)
for (i = 0; i < 2834; i++)
baz ();
}
void
foo2 ()
{
int i;
#pragma omp parallel for schedule (static, 16)
for (i = 0; i < 2834; i++)
for (;;)
;
}
void
bar2 ()
{
int i;
#pragma omp parallel for schedule (static, 16)
for (i = 0; i < 2834; i++)
baz ();
}
