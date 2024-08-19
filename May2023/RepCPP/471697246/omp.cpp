#include "fonctions.hpp"

int main(int argc, char **argv) {

int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

int pid, nprocs;
MPI_Comm_rank(MPI_COMM_WORLD, &pid);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

assert(argc == 5);

const int n = atoi(argv[1]);
const int m = atoi(argv[2]);
const int root = atoi(argv[3]);
const std::string name = argv[4];

const size_t nn = n * n;
const size_t mn = m * n;
const size_t data_size = sizeof(int);
const int nb_zero = (int) (0.1 * n);

Time debut;

int *matrix = new int[nn];
int *vectors;

std::fstream f;
if (pid == root) {
srand(time(nullptr));
for (size_t i = 0; i < nn; i++) matrix[i] = rand() % 20;

LOG(
std::clog << "Matrix : " << std::endl;
for (size_t i = 0; i < n; ++i) {
std::clog << "\t";
for (size_t j = 0; j < n; ++j) std::clog << matrix[i * n + j] << " ";
std::clog << std::endl;
}
)

vectors = new int[mn];
for (size_t i = 0; i < m; i++) generate_vector(n, vectors + (i * n), rand() % nb_zero);

LOG(
std::clog << "Vectors : " << std::endl;
for (size_t i = 0; i < m; ++i) {
std::clog << "\tVector " << i <<" [ ";
for (size_t j = 0; j < n; ++j) std::clog << vectors[i * n + j] << " ";
std::clog << "]" << std::endl;
}
)
}
if (pid == root) debut = NOW;

MPI_Bcast(matrix, nn, MPI_INT, root, MPI_COMM_WORLD);

int sendcount[nprocs];
int displs[nprocs];
int counter = 0;
const int qte = (m / nprocs);
const int rst = m % nprocs;
for (int i = 0; i < nprocs; ++i) {
displs[i] = counter;
int x = (rst > i) ? (qte + 1) * n : qte * n;
sendcount[i] = x;
counter += x;
}
const size_t sz = sendcount[pid] / n;
int *batch = new int[sendcount[pid]];

MPI_Scatterv(vectors, sendcount, displs, MPI_INT,
batch, sendcount[pid], MPI_INT,
root, MPI_COMM_WORLD);

LOG(
for (size_t i = 0; i < sz; ++i) {
std::clog << "PID: " << pid << "; Vector " << i <<" [ ";
for (size_t j = 0; j < n; ++j) std::clog << batch[i * n + j] << " ";
std::clog << "]" << std::endl;
}
)


#pragma omp parallel for
for (size_t i = 0; i < sz; ++i) {
int result[n];
matrix_vector_omp(n, matrix, batch + (i * n), result);
memcpy(batch + (i * n), result, n * data_size);
}

MPI_Gatherv(batch, sendcount[pid], MPI_INT,
vectors, sendcount, displs, MPI_INT,
root, MPI_COMM_WORLD);

if (pid == root) {
std::chrono::duration<double> elapsed_seconds = NOW - debut;
std::cout << "Time: " << elapsed_seconds.count() << "s" << std::endl;
}

if (pid == root) {
f.open(name, std::fstream::out);

f << "Result:" << std::endl;
for (size_t i = 0; i < m; i++) {
f << "\tVector " << i << " [ ";
for (size_t j = 0; j < n; j++) f << vectors[i * n + j] << " ";
f << "]" << std::endl;
}
f.close();

LOG(
std::clog << "Result:" << std::endl;
for (size_t i = 0; i < m; i++) {
std::clog << "\tVector " << i << " [ ";
for (size_t j = 0; j < n; j++) std::clog << vectors[i * n + j] << " ";
std::clog << "]" << std::endl;
}
)
}

MPI_Finalize();

delete[] matrix;
delete[] batch;
if (pid == root) delete[] vectors;

return 0;
}