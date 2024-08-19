int N = 0;
struct R {
int a;
R() : a(0) {N++;}
};
int main(int argc, char** argv)
{
#pragma analysis_check assert_decl upper_exposed(N) defined(N, r)
struct R r;
return 0;
}
