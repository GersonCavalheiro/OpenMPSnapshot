#pragma interface
struct T
{
T *p;
void baz ()
{
p->baz ();
}
};
void foo (T *p)
{
p->baz ();
}
