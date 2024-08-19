class a
{
bool b ();
};
bool
a::b ()
{
#pragma omp parallel
;
return true;
}
