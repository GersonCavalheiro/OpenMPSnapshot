#pragma pack(1)
struct
{
float f0;
} a;
extern void foo (int);
int
main (void)
{
for (;;)
foo ((int) a.f0);
}
