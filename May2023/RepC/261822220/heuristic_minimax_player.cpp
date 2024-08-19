#include "player.h"
#include "game_logic.h"
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <mutex>  
using namespace player;
using namespace std::chrono;
bool maxing_cmp(const edge_t &a, const edge_t &b)
{
return a.value>b.value;
}
bool mining_cmp(const edge_t &a, const edge_t &b)
{
return a.value<b.value;
}
const int EXP_B[7] = {100,80,60,40,20,10,5};
const double offensivity_factor = 1.5;
const double fallen_factor = 2.0;
const double edges_factor = 0.2;
const double centrality_factor = 1.0;
const double co_factor = 0.03;
const double adj_factor = 0.1;
typedef struct {
double alpha;
double beta;
} alpha_beta;
volatile alpha_beta ab = {-1000000000.0, 1000000000.0};
alpha_beta mpi_ab = {-1000000000.0, 1000000000.0};
int mpi_call_cnt = 0;
#if MPI
MPI_Request receive_alpha_req;
#endif
std::mutex mtx; 
double HeuristicMinimaxPlayer::heuristic_value(state_t* state)
{
if(is_terminal(state))
return 100*(get_winner(state)==player_col?1:-1);
int nB[2] = {14-state->fallen_marbles[0],14-state->fallen_marbles[1]};
double rst = 0;
rst+= fallen_factor * (offensivity_factor * state->fallen_marbles[adv_col] - state->fallen_marbles[player_col]);
rst+= edges_factor * (state->n_marbles_on_edge[adv_col] - state->n_marbles_on_edge[player_col]);
rst+= centrality_factor * (state->centrality[adv_col]*1.0/nB[adv_col] - state->centrality[player_col]*1.0/nB[player_col]);
rst+= co_factor* (state->occupying_center[player_col]-state->occupying_center[adv_col]);
rst+= adj_factor* (state->n_adjacent_pairs[player_col]-state->n_adjacent_pairs[adv_col]);
return rst;
}
double HeuristicMinimaxPlayer::move_heuristic_value(state_t* state, move_t* move)
{
char fm[2];
char nmoe[2];
char c[2];
char nap[2];
char oc[2];
for(int i=0;i<2;i++)
{
fm[i] = state->fallen_marbles[i]+move->delta_fm[i];
nmoe[i] = state->n_marbles_on_edge[i]+move->delta_nmoe[i];
c[i] = state->centrality[i]+move->delta_c[i];
nap[i] = state->n_adjacent_pairs[i]+move->delta_nap[i];
oc[i] = state->occupying_center[i]+move->delta_oc[i];
}
int nB[2] = {14-fm[0],
14-fm[1]};
double rst = 0;
rst+= fallen_factor * (offensivity_factor * fm[adv_col] - fm[player_col]);
rst+= edges_factor * (nmoe[adv_col] - nmoe[player_col]);
rst+= centrality_factor * (c[adv_col]*1.0/nB[adv_col] - c[player_col]*1.0/nB[player_col]);
rst+= co_factor* (oc[player_col]-oc[adv_col]);
rst+= adj_factor* (nap[player_col]-nap[adv_col]);
return rst;
}
HeuristicMinimaxPlayer::HeuristicMinimaxPlayer()
{
init_depth = 6;
time_budget = 5.0;
}
HeuristicMinimaxPlayer::HeuristicMinimaxPlayer(int init_d, double t_budget)
{
init_depth = init_d;
time_budget = t_budget;
}
inline bool done(time_point<system_clock> start_time, double time_budget)
{
auto t = system_clock::now();
return duration_cast<milliseconds>(t-start_time).count()>=1000*time_budget;
}
return_t HeuristicMinimaxPlayer::process(state_t* state, double alpha, double beta, int depth, int MAX_DEPTH)
{
bool maxing = state->next_player == player_col;
return_t rst;
if(is_terminal(state))
{
rst.value = 100*(get_winner(state)==player_col?1:-1);
rst.move_id = -1;
return rst;
}
if(depth==MAX_DEPTH)
{
rst.value = heuristic_value(state);
rst.move_id = -1;
return rst;
}
std::vector<move_t*> moves;
possible_moves(state, moves);
std::vector<edge_t> edges;
for(int i=0;i<moves.size();i++)
{
edge_t e;
e.move=moves[i];
e.value=move_heuristic_value(state, e.move);
edges.push_back(e);
}
if(maxing)
sort(edges.begin(),edges.end(),maxing_cmp);
else
sort(edges.begin(),edges.end(),mining_cmp);
int BREADTH = depth<7?EXP_B[depth]:5;
rst.value = maxing?-100000:100000;
auto start_explore = high_resolution_clock::now();
for(int i=0;i<BREADTH && i < edges.size();i++)
{
apply_move(state, edges[i].move);
return_t ch = process(state, alpha, beta, depth+1, MAX_DEPTH);
if(determined_move!=-1 && done(start_time, time_budget))
{
compute_interrupted=true;
return rst;
}
reverse_move(state, edges[i].move);
if(maxing)
{
if(ch.value>rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
if(rst.value>alpha)
alpha=rst.value;
if(beta<=alpha)
break;
}
}
else
{
if(ch.value<rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
if(rst.value<beta)
beta=rst.value;
if(beta<=alpha)
break;
}
}
}
for(int i=0;i<edges.size();i++)
free(edges[i].move);
return rst;
}
#if OMP
return_t HeuristicMinimaxPlayer::process_omp(state_t* state, double alpha, double beta, int depth, int MAX_DEPTH)
{
move_t moves[BUFFER_SIZE];
edge_t edges[BUFFER_SIZE];
return_t rst;
bool maxing = state->next_player == player_col;
if(alpha<ab.alpha)
{
alpha=ab.alpha;
if(beta<=alpha)
return rst;
}
if(is_terminal(state))
{
rst.value = 100*(get_winner(state)==player_col?1:-1);
rst.move_id = -1;
return rst;
}
if(depth==MAX_DEPTH)
{
rst.value = heuristic_value(state);
rst.move_id = -1;
return rst;
}
int num_possible_moves = possible_moves_omp(state, moves);
for(int i=0;i<num_possible_moves;i++)
{
edges[i].move=&moves[i];
edges[i].value=move_heuristic_value(state, edges[i].move);
}
if(maxing)
std::sort(edges, edges+num_possible_moves, maxing_cmp);
else
std::sort(edges, edges+num_possible_moves, mining_cmp); 
int BREADTH = depth<7?EXP_B[depth]:5;
rst.value = maxing?-100000:100000;
for(int i=0;i<BREADTH && i < num_possible_moves;i++)
{
apply_move(state, edges[i].move);
return_t ch = process_omp(state, alpha, beta, depth+1, MAX_DEPTH);
if(determined_move!=-1 && done(start_time, time_budget))
{
compute_interrupted=true;
return rst;
}
reverse_move(state, edges[i].move);
if(alpha<ab.alpha)
alpha=ab.alpha;
if(maxing)
{
if(ch.value>rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
if(rst.value>alpha)
alpha=rst.value;
if(beta<=alpha){
break;
}
}
}
else
{
if(ch.value<rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
if(rst.value<beta)
beta=rst.value;
if(beta<=alpha){
break;
}
}
}
}
return rst;
}
#endif
#if MPI
return_t HeuristicMinimaxPlayer::process_mpi(state_t* state, double alpha, double beta, int depth, int MAX_DEPTH)
{
mpi_call_cnt++;
if(mpi_call_cnt == 10){
mpi_call_cnt = 0;
receive_updated_alpha(&mpi_ab.alpha, &receive_alpha_req);
}
move_t moves[BUFFER_SIZE];
edge_t edges[BUFFER_SIZE];
bool maxing = state->next_player == player_col;
if(alpha<mpi_ab.alpha)
alpha=mpi_ab.alpha;
return_t rst;
if(is_terminal(state))
{
rst.value = 100*(get_winner(state)==player_col?1:-1);
rst.move_id = -1;
return rst;
}
if(depth==MAX_DEPTH)
{
rst.value = heuristic_value(state);
rst.move_id = -1;
return rst;
}
int num_possible_moves = possible_moves_omp(state, moves);
for(int i=0;i<num_possible_moves;i++)
{
edges[i].move=&moves[i];
edges[i].value=move_heuristic_value(state, edges[i].move);
}
if(maxing)
std::sort(edges, edges+num_possible_moves, maxing_cmp);
else
std::sort(edges, edges+num_possible_moves, mining_cmp); 
int BREADTH = depth<7?EXP_B[depth]:5;
rst.value = maxing?-100000:100000;
for(int i=0;i<BREADTH && i < num_possible_moves;i++)
{
apply_move(state, edges[i].move);
return_t ch = process_mpi(state, alpha, beta, depth+1, MAX_DEPTH);
if(determined_move!=-1 && done(start_time, time_budget))
{
compute_interrupted=true;
return rst;
}
reverse_move(state, edges[i].move);
if(maxing)
{
if(ch.value>rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
if(rst.value>alpha)
alpha=rst.value;
if(beta<=alpha){
break;
}
}
}
else
{
if(ch.value<rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
if(rst.value<beta)
beta=rst.value;
if(beta<=alpha){
break;
}
}
}
}
return rst;
}
#endif
move_t *find_move_by_id(std::vector<edge_t> &edges, int threshold, int move_id){
for(int i=0; i<threshold; i++){
if(edges[i].move->move_id == move_id){
return edges[i].move;
}
}
move_t* move = {};
move->move_id = -1;
return move;
}
std::pair<int, double> HeuristicMinimaxPlayer::next_move(state_t* state, move_t* prev_move)
{
player_col = state->next_player;
adv_col = WHITE + BLACK - player_col;
return_t rst;
start_time = system_clock::now();
determined_move = -1;
compute_interrupted = false;
#if OMP
state_t first_layer_states[EXP_B[0]];
return_t first_layer_results[EXP_B[0]];
bool maxing = state->next_player == player_col;
int depth=init_depth-1;
do
{
depth++;
std::vector<move_t*> moves;
possible_moves(state, moves);
std::vector<edge_t> edges;
return_t rst;
int x;
for(int i=0;i<moves.size();i++)
{
edge_t e;
e.move=moves[i];
e.value=move_heuristic_value(state, e.move);
edges.push_back(e);
}
if(maxing)
sort(edges.begin(),edges.end(),maxing_cmp);
else
sort(edges.begin(),edges.end(),mining_cmp);
int threshold = EXP_B[0] <= edges.size() ? EXP_B[0] : edges.size();
for(int x=0; x<threshold; x++){
clone_state(&first_layer_states[x], state);
apply_move(&first_layer_states[x], edges[x].move);
}
ab.alpha = -1000000000.0;
ab.beta = 1000000000.0;
#pragma omp parallel for schedule(dynamic,1) private(x)
for(int x=0;x<threshold;x++){
first_layer_results[x] = process_omp(&first_layer_states[x], ab.alpha, ab.beta, 1, depth);
mtx.lock();
double v = first_layer_results[x].value;
if(v>ab.alpha)
ab.alpha=v;
mtx.unlock();
}
if(compute_interrupted)
depth--;
else
{
rst.value = maxing?-100000:100000;
for(int i=0;i<threshold;i++)
{
return_t ch = first_layer_results[i];
if(maxing)
{
if(ch.value>rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
}
}
else
{
if(ch.value<rst.value)
{
rst.value=ch.value;
rst.move_id = edges[i].move->move_id;
}
}
}
determined_move=rst.move_id;
}
for(int i=0;i<edges.size();i++)
free(edges[i].move);
} while(!done(start_time, time_budget));
return std::pair<int, double>(determined_move, depth);
#elif MPI
mpi_ab.alpha = -1000000000.0;
mpi_ab.beta = 1000000000.0;
int this_process = 0;
int process_count = 1;
MPI_Comm_size(MPI_COMM_WORLD, &process_count);
MPI_Comm_rank(MPI_COMM_WORLD, &this_process);
bool mpi_master = this_process == 0;
bool maxing = state->next_player == player_col;
int depth=init_depth-1;
std::vector<move_t*> moves;
possible_moves(state, moves);
MPI_Request *send_reqs;
MPI_Request *recv_reqs;
int *process_assigned_move;
return_t *results;
double *results_buffer;
if(mpi_master){
send_reqs = (MPI_Request*) malloc(process_count*sizeof(MPI_Request));
recv_reqs = (MPI_Request*) malloc(process_count*sizeof(MPI_Request));
process_assigned_move = (int*) malloc(process_count*sizeof(int));
results = (return_t*) malloc(process_count*sizeof(return_t));
results_buffer = (double*)malloc(process_count*sizeof(double)*2);
}
do{
std::vector<edge_t> edges;
for(int i=0;i<moves.size();i++)
{
edge_t e;
e.move=moves[i];
e.value=move_heuristic_value(state, e.move);
edges.push_back(e);
}
if(maxing)
sort(edges.begin(),edges.end(),maxing_cmp);
else
sort(edges.begin(),edges.end(),mining_cmp);
int threshold = EXP_B[0] <= edges.size() ? EXP_B[0] : edges.size();
if(mpi_master){
depth++;
int finished_cnt=0;
rst.value = maxing?-100000:100000;
for(int i=1; i<process_count; i++){
process_assigned_move[i] = assign_one_move(edges, i, results_buffer+2*i, &send_reqs[i], &recv_reqs[i]);
}
rst.move_id = edges[0].move->move_id;
while(finished_cnt < threshold){ 
for(int i=1; i<process_count; i++){
int assigned_move = process_assigned_move[i];
if(assigned_move == -1) continue;
bool finished = receive_one_finished_result(edges, i, results_buffer+2*i,  &results[i], &send_reqs[i], &recv_reqs[i], &process_assigned_move[i]);
if(finished){
finished_cnt++;
if(results[i].move_id){
compute_interrupted = true;
}else{
if(results[i].value > mpi_ab.alpha){
mpi_ab.alpha = results[i].value;
update_alpha(mpi_ab.alpha, process_count);
}
if(maxing)
{
if(results[i].value>rst.value)
{
rst.value=results[i].value;
rst.move_id = assigned_move;
}
}
else
{
if(results[i].value<rst.value)
{
rst.value=results[i].value;
rst.move_id = assigned_move;
}
}
determined_move=rst.move_id;
}
}
}
}
send_final_result(send_reqs, process_count, determined_move);
}else{
depth++;
mpi_call_cnt = 0;
MPI_Irecv(&mpi_ab.alpha, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &receive_alpha_req);
int move_id  = receive_one_move();
while(move_id != -1){
move_t* move = find_move_by_id(edges, threshold, move_id);
state_t state_c;
clone_state(&state_c, state);
apply_move(&state_c, move);
return_t result = process_mpi(&state_c, mpi_ab.alpha, mpi_ab.beta, 1, depth);
send_one_finished_result(result.value, compute_interrupted);
move_id  = receive_one_move();
}
determined_move = receive_final_result();
}
if(compute_interrupted){
depth--;
} 
MPI_Barrier(MPI_COMM_WORLD);
}while(!done(start_time, time_budget));
for(int i=0;i<moves.size();i++){
free(moves[i]);
}
return std::pair<int, double>(determined_move, depth);
#else
int depth=init_depth-1;
do
{
depth++;
int move_id =  process(state, -1000000000.0, 1000000000.0, 0, depth).move_id;
if(compute_interrupted)
depth--;
else
determined_move=move_id;
} while(!done(start_time, time_budget));
return std::pair<int, double>(determined_move, depth);
#endif 
}