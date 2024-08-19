



/
ll_t fib(ll_t n)
{
if (n <= 1) {  
return n;  
}
else {
ll_t x = fib(n - 1);
ll_t y = fib(n - 2);

return x + y;
}
}




ll_t p_fib(ll_t n)
{
ll_t x, y;
if (n <= 1) {  
return n;  
}
else {
#pragma omp task shared(x) firstprivate(n)
x = p_fib(n - 1);                   
#pragma omp task shared(y) firstprivate(n)
y = p_fib(n - 2);                   
#pragma omp taskwait
return x + y;                       
}
}

int main()
{

int n = 20;
int fibn;
#ifdef __PARALLEL__
omp_set_dynamic(0);
omp_set_num_threads(4);
auto start = std::chrono::system_clock::now();
#pragma omp parallel shared(n)
{
#pragma omp single
fibn = p_fib(n);
}
auto end = std::chrono::system_clock::now();
auto dur = end - start;
auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
std::cout << "parallel fib(" << n << ")=" << fibn << std::endl;
std::cout << msec << " milli sec\n";

#else
auto start = std::chrono::system_clock::now();
fibn = fib(n);;
auto end = std::chrono::system_clock::now();
auto dur = end - start;
auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
std::cout << "fib(" << n << ")=" << fibn << std::endl;
std::cout << msec << " milli sec\n";
#endif

getchar();
return 0;
}

