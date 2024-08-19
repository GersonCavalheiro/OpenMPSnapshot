#pragma once

namespace monolish {

namespace mpi {


class comm {
private:

MPI_Comm my_comm = 0;
comm(){};
~comm(){};

std::vector<MPI_Request> requests;

public:
comm(const comm &) = delete;
comm &operator=(const comm &) = delete;
comm(comm &&) = delete;
comm &operator=(comm &&) = delete;

static comm &get_instance() {
static comm instance;
return instance;
}


void Init();


void Init(int argc, char **argv);


bool Initialized() const;



[[nodiscard]] MPI_Comm get_comm() const { return my_comm; }


void set_comm(MPI_Comm external_comm);


void Finalize();


[[nodiscard]] int get_rank();


[[nodiscard]] int get_size();



void Barrier() const;


void Send(double val, int dst, int tag) const;


void Send(float val, int dst, int tag) const;


void Send(int val, int dst, int tag) const;


void Send(size_t val, int dst, int tag) const;


void Send(std::vector<double> &vec, int dst, int tag) const;


void Send(std::vector<float> &vec, int dst, int tag) const;


void Send(std::vector<int> &vec, int dst, int tag) const;


void Send(std::vector<size_t> &vec, int dst, int tag) const;


void Send(monolish::vector<double> &vec, int dst, int tag) const;


void Send(monolish::vector<float> &vec, int dst, int tag) const;


MPI_Status Recv(double val, int src, int tag) const;


MPI_Status Recv(float val, int src, int tag) const;


MPI_Status Recv(int val, int src, int tag) const;


MPI_Status Recv(size_t val, int src, int tag) const;


MPI_Status Recv(std::vector<double> &vec, int src, int tag) const;


MPI_Status Recv(std::vector<float> &vec, int src, int tag) const;


MPI_Status Recv(std::vector<int> &vec, int src, int tag) const;


MPI_Status Recv(std::vector<size_t> &vec, int src, int tag) const;


MPI_Status Recv(monolish::vector<double> &vec, int src, int tag) const;


MPI_Status Recv(monolish::vector<float> &vec, int src, int tag) const;


void Isend(double val, int dst, int tag);


void Isend(float val, int dst, int tag);


void Isend(int val, int dst, int tag);


void Isend(size_t val, int dst, int tag);


void Isend(const std::vector<double> &vec, int dst, int tag);


void Isend(const std::vector<float> &vec, int dst, int tag);


void Isend(const std::vector<int> &vec, int dst, int tag);


void Isend(const std::vector<size_t> &vec, int dst, int tag);


void Isend(const monolish::vector<double> &vec, int dst, int tag);


void Isend(const monolish::vector<float> &vec, int dst, int tag);


void Irecv(double val, int src, int tag);


void Irecv(float val, int src, int tag);


void Irecv(int val, int src, int tag);


void Irecv(size_t val, int src, int tag);


void Irecv(std::vector<double> &vec, int src, int tag);


void Irecv(std::vector<float> &vec, int src, int tag);


void Irecv(std::vector<int> &vec, int src, int tag);


void Irecv(std::vector<size_t> &vec, int src, int tag);


void Irecv(monolish::vector<double> &vec, int src, int tag);


void Irecv(monolish::vector<float> &vec, int src, int tag);


void Waitall();


[[nodiscard]] double Allreduce(double val) const;


[[nodiscard]] float Allreduce(float val) const;


[[nodiscard]] int Allreduce(int val) const;


[[nodiscard]] size_t Allreduce(size_t val) const;


[[nodiscard]] double Allreduce_sum(double val) const;


[[nodiscard]] float Allreduce_sum(float val) const;


[[nodiscard]] int Allreduce_sum(int val) const;


[[nodiscard]] size_t Allreduce_sum(size_t val) const;


[[nodiscard]] double Allreduce_prod(double val) const;


[[nodiscard]] float Allreduce_prod(float val) const;


[[nodiscard]] int Allreduce_prod(int val) const;


[[nodiscard]] size_t Allreduce_prod(size_t val) const;


[[nodiscard]] double Allreduce_max(double val) const;


[[nodiscard]] float Allreduce_max(float val) const;


[[nodiscard]] int Allreduce_max(int val) const;


[[nodiscard]] size_t Allreduce_max(size_t val) const;


[[nodiscard]] double Allreduce_min(double val) const;


[[nodiscard]] float Allreduce_min(float val) const;


[[nodiscard]] int Allreduce_min(int val) const;


[[nodiscard]] size_t Allreduce_min(size_t val) const;


void Bcast(double &val, int root) const;


void Bcast(float &val, int root) const;


void Bcast(int &val, int root) const;


void Bcast(size_t &val, int root) const;


void Bcast(monolish::vector<double> &vec, int root) const;


void Bcast(monolish::vector<float> &vec, int root) const;


void Bcast(std::vector<double> &vec, int root) const;


void Bcast(std::vector<float> &vec, int root) const;


void Bcast(std::vector<int> &vec, int root) const;


void Bcast(std::vector<size_t> &vec, int root) const;


void Gather(monolish::vector<double> &sendvec,
monolish::vector<double> &recvvec, int root) const;


void Gather(monolish::vector<float> &sendvec,
monolish::vector<float> &recvvec, int root) const;


void Gather(std::vector<double> &sendvec, std::vector<double> &recvvec,
int root) const;


void Gather(std::vector<float> &sendvec, std::vector<float> &recvvec,
int root) const;


void Gather(std::vector<int> &sendvec, std::vector<int> &recvvec,
int root) const;


void Gather(std::vector<size_t> &sendvec, std::vector<size_t> &recvvec,
int root) const;


void Scatter(monolish::vector<double> &sendvec,
monolish::vector<double> &recvvec, int root) const;


void Scatter(monolish::vector<float> &sendvec,
monolish::vector<float> &recvvec, int root) const;


void Scatter(std::vector<double> &sendvec, std::vector<double> &recvvec,
int root) const;


void Scatter(std::vector<float> &sendvec, std::vector<float> &recvvec,
int root) const;


void Scatter(std::vector<int> &sendvec, std::vector<int> &recvvec,
int root) const;


void Scatter(std::vector<size_t> &sendvec, std::vector<size_t> &recvvec,
int root) const;
};

} 
} 