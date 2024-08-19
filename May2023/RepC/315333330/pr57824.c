void bar ();
void foo ()
{
#pragma omp parallel num_threads(sizeof R"(
abc
)")
bar ();
}
