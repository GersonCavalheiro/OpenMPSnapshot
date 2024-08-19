#pragma once

#include "pipeline/schedule.hpp"
#include "net/mpihead.hpp"
#include "net/epa_mpi_util.hpp"
#include "util/logging.hpp"
#include "util/stringify.hpp"
#include "util/Timer.hpp"

#ifdef __MPI
#include <mpi.h>



class Intercom
{
public:
explicit Intercom( size_t const num_stages )
{
std::vector<double> initial_difficulty(num_stages, 1.0);
MPI_Comm_rank(MPI_COMM_WORLD, &local_rank_);
MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

auto init_nps = solve(num_stages, world_size_, initial_difficulty);
assign(local_rank_, init_nps, schedule_, &local_stage_);

LOG_DBG << "Schedule: " << stringify(schedule_);
}

Intercom()   = delete;
~Intercom()  = default;


auto& schedule( size_t const id )
{
return schedule_[id];
}


auto& previous_requests()
{
return prev_requests_;
}


bool stage_active(const size_t stage_id) const
{
return stage_id == static_cast<size_t>(local_stage_);
}


void rebalance(Timer<>& timer)
{
MPI_BARRIER(MPI_COMM_WORLD);

LOG_DBG << "Rebalancing...";
const auto foreman = schedule_[local_stage_][0];
const auto num_stages = schedule_.size();
Timer<> per_node_avg(timer.avg_duration());
LOG_DBG1 << "aggregate the runtime statistics...";
Timer<> dummy;
epa_mpi_gather(per_node_avg, foreman, schedule_[local_stage_], local_rank_, dummy);
LOG_DBG1 << "Runtime aggregate done!";

std::vector<double> perstage_total(num_stages);

int color = (local_rank_ == foreman) ? 1 : MPI_UNDEFINED;
MPI_Comm foreman_comm;
MPI_Comm_split(MPI_COMM_WORLD, color, local_stage_, &foreman_comm);

if (local_rank_ == foreman)
{
double total_stagetime = per_node_avg.sum();
LOG_DBG1 << "Foremen allgather...";
MPI_Allgather(&total_stagetime, 1, MPI_DOUBLE, &perstage_total[0], 1, MPI_DOUBLE, foreman_comm);
LOG_DBG1 << "Foremen allgather done!";
MPI_Comm_free(&foreman_comm);
}
epa_mpi_waitall(prev_requests_);

MPI_BARRIER(MPI_COMM_WORLD);
LOG_DBG1 << "Broadcasting...";
MPI_Comm stage_comm;
MPI_Comm_split(MPI_COMM_WORLD, local_stage_, local_rank_, &stage_comm);

int split_key = local_rank_ == foreman ? -1 : local_rank_;
MPI_Comm_split(MPI_COMM_WORLD, local_stage_, split_key, &stage_comm);
MPI_Bcast(&perstage_total[0], num_stages, MPI_DOUBLE, 0, stage_comm);
MPI_Comm_free(&stage_comm);
LOG_DBG1 << "Broadcasting done!";

LOG_DBG1 << "perstage total: " << stringify(perstage_total);

to_difficulty(perstage_total);

LOG_DBG1 << "perstage difficulty: " << stringify(perstage_total);

auto nps = solve(num_stages, world_size_, perstage_total);
reassign(local_rank_, nps, schedule_, &local_stage_);
LOG_DBG1 << "New Schedule: " << stringify(nps);

LOG_DBG << "Rebalancing done!";

prev_requests_.clear();
timer.clear();
}

void barrier() const
{
MPI_BARRIER(MPI_COMM_WORLD);
}

int rank()
{
return local_rank_;
}

private:
int local_rank_   = -1;
int world_size_   = -1;
int local_stage_  = -1;

schedule_type schedule_;
previous_request_storage_t prev_requests_;

};

#else

class Intercom
{
public:
explicit Intercom(const size_t) {}
Intercom() = default;
~Intercom()= default;


bool stage_active(const size_t) const { return true; }
void rebalance(Timer<>&) { }
void barrier() const { }
int rank() { return 0; }

};

#endif 
