const int N = 5;
void AddOne( int &y )
{
int y_;
#pragma analysis_check assert upper_exposed(y, N) defined(y, y_)
if( y > 50 )
return;
else
{
y_ = y++ + N;
AddOne( y_ );
}
}
int main()
{
int x = 5;
#pragma analysis_check assert upper_exposed(x, N) defined(x)
AddOne( x );
return 0;
}