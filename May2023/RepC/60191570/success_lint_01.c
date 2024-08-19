int main() {
int x, y, z = 0;
#pragma oss lint in(x) out(y) inout(z)
{
x;
y = 2;
z++;
}
}
