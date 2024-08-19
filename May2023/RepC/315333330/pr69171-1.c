#pragma pack(2)
struct {
int n;
} a;
extern void fn2 (float);
void
fn1 ()
{
fn2 (a.n);
}
