#pragma pack(1)
struct S
{
char a;
};
int test[sizeof(struct S) == 1 ? 1: -1];
