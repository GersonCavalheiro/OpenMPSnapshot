#pragma GCC target "+nosve"
unsigned char foo(const unsigned char *buffer, unsigned int length)
{
unsigned char sum;
unsigned int  count;
for (sum = 0, count = 0; count < length; count++) {
sum = (unsigned char) (sum + *(buffer + count));
}
return sum;
}
