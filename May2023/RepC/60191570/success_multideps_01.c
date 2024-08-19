void bar()
{
int v[10][10];
#pragma oss task inout({v[i][j], i=0:9, j=0:i})
{}
}
