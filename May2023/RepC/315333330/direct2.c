#define HASH #
#define HASHDEFINE #define
#define HASHINCLUDE #include
HASH include "somerandomfile" 
int resync_parser_1; 
HASHINCLUDE <somerandomfile> 
int resync_parser_2;
void g1 ()
{
HASH define X 1 
int resync_parser_3;
}
void g2 ()
{
HASHDEFINE  Y 1 
int resync_parser_4;
}
#pragma GCC dependency "direct2.c"
#
void f ()
{
int i = X;    
int j = Y;    
}
#define slashstar /##*
#define starslash *##/
slashstar starslash 
