void AddOne( int &y )
{
y++;
}
int main()
{
int x = 5;
#pragma analysis_check assert upper_exposed(x) defined(x)
AddOne( x );
return 0;
}