template <class T>
struct S
{
S (const T &);
~S ();
T t;
};
template <class T>
S<T>::S (const T &x)
{
t = x;
}
template <class T>
S<T>::~S ()
{
}
#pragma GCC visibility push(hidden)
struct U
{
S<int> s;
U () : s (6) { }
} u;
#pragma GCC visibility pop
