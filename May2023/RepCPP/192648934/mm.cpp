



/
mat_t p_square_matrix_multiply(const mat_t& A, const mat_t& B)
{
int n = A.size();
mat_t C(n, vec_t(n));

#pragma omp parallel shared(A, B, C, n)
{
int i, j;
#pragma omp for collapse(2) private(i, j)
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++) {
C[i][j] = 0;
for (int k = 0; k < n; k++) {
C[i][j] = C[i][j] + A[i][k] * B[k][j]; 
}
}
}
}

return C;
}


int main()
{   
const int N = 8;

mat_t A(N, vec_t(N));
mat_t B(N, vec_t(N));

std::cout << "A = ["; 
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
A[i][j] = 2;
std:: cout << A[i][j] << " ";
}
std::cout << std::endl;
}
std::cout << "]" << std::endl;

std::cout << "B = ["; 
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
B[i][j] = 2;
std:: cout << B[i][j] << " ";
}
std::cout << std::endl;
}
std::cout << "]" << std::endl;



omp_set_dynamic(0);
omp_set_num_threads(4);


mat_t C;
C = p_square_matrix_multiply(A, B);


std::cout << "C = ["; 
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
std:: cout << C[i][j] << " ";
}
std::cout << std::endl;
}
std::cout << "]" << std::endl;


return 0;
}
