struct unpacked
{
int i;
char c;
};
#pragma pack(1)
struct packed
{
int i;
char c;
};
struct packed_contains_unpacked
{
char c;
struct unpacked uuuu;  
}; 			
union contains_unpacked
{
char c;
struct unpacked uuuu;  
};
struct packed_contains_packed
{
char c;
struct packed ppppp;   
};
#pragma pack()
struct unpacked_contains_packed
{
char c;
struct packed p;
};
struct unpacked_contains_unpacked
{
char c;
struct unpacked u;
};
int s1 = sizeof (struct unpacked);
int s2 = sizeof (struct packed);
int s3 = sizeof (struct packed_contains_unpacked);
int s4 = sizeof (struct packed_contains_packed);
int s5 = sizeof (struct unpacked_contains_packed);
int s6 = sizeof (struct unpacked_contains_unpacked);
