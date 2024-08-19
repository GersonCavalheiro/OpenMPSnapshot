extern "C" int printf (const char *, ...);
#pragma pack(4)
template <typename X >
struct T
{
char      x1;   
int       x2;
};
template <class T>
int f()
{
struct A { char i1; int i2; };
return sizeof (A);
}
#pragma pack(1)
template struct T<int>;   
template int f<int>();
#pragma pack(4)
template struct T<float>; 
template int f<double>();
int main()
{
printf("sizeof T<int>   = %d\n", sizeof(T<int>));
printf("sizeof T<float> = %d\n", sizeof(T<float>));
printf("f<int>()        = %d\n", f<int>());
printf("f<float>()      = %d\n", f<float>());
return (sizeof(T<int>) != sizeof(T<float>)
|| f<int>() != f<float>());
}
