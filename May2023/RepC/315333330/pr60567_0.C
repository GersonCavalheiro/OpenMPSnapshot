#pragma implementation
struct S {};
#pragma interface
struct T
{
virtual void foo (const S &) = 0;
};
struct U
{
virtual void bar (const S &) = 0;
};
struct V : public T, public U
{
virtual void bar (const S &) {}
};
