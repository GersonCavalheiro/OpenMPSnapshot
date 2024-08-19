#pragma pack(1)
struct {
signed : 1;
unsigned long a;
} b;
void c(void)
{
b.a = 0;
for (; b.a <= 45; b.a = (long)b.a + 1)
;
}
