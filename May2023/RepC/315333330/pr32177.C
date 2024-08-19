struct A
{
A () {}
~A () {}
int s () const { return 1; }
};
void
f1 ()
{
#pragma omp parallel for
for (int i = 1; i <= A ().s (); ++i)
;
}
void
f2 ()
{
#pragma omp parallel for
for (int i = A ().s (); i <= 20; ++i)
;
}
void
f3 ()
{
#pragma omp parallel for
for (int i = 1; i <= 20; i += A ().s ())
;
}
void
f4 ()
{
int i;
#pragma omp parallel for
for (i = A ().s (); i <= 20; i++)
;
}
