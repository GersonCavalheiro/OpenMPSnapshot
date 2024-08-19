#include "player.h"
#include "game_logic.h"
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <cmath>
#include <chrono>
#include <mutex>  
#define C_PUCT 2.0
#define FACTOR 0.75
using namespace std;
using namespace player;
using namespace chrono;
int nodes_counter =0;
MCTS_Node* new_node()
{
nodes_counter++;
MCTS_Node* rst = (MCTS_Node*)calloc(1, sizeof(MCTS_Node));
return rst;
}
const double offensivity_factor = 1.5;
const double fallen_factor = 2.0;
const double edges_factor = 0.2;
const double centrality_factor = 1.0;
const double co_factor = 0.3;
const double adj_factor = 0.1;
bool sorting_cmp(const edge_t &a, const edge_t &b)
{
return a.value<b.value;
}
double heuristic_value(state_t* state)
{
int player_col = BLACK+WHITE-state->next_player;
int adv_col = state->next_player;
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
double move_heuristic_value(state_t* state, move_t* move)
{
int player_col = BLACK+WHITE-state->next_player;
int adv_col = state->next_player;
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
rst+= adj_factor* (nap[player_col]-nap[adv_col]);
return rst;
}
MCTSPlayer::MCTSPlayer()
{
time_budget = 5.0;
max_rollouts=1000000000;
gen.seed(0);
root = NULL;
}
MCTSPlayer::MCTSPlayer(int max_r, double t_budget, int seed)
{
time_budget = t_budget;
max_rollouts = max_r;
gen.seed(seed);
root = NULL;
}
inline double get_time()
{
return clock() / (double) CLOCKS_PER_SEC;
}
inline MCTS_Node* select(MCTS_Node* root, int player_col)
{
MCTS_Node* node = root;
while(1)
{
#if OMP
node->mtx.lock();
#endif
if(!node->expanded)
break;
double best = -1000000;
int index = -1;
int sum_N = 0;
for(int m =0;m<node->n_children;m++)
sum_N +=node->children[m].N_effective;
double sqrt_N = sqrt(sum_N);
for(int m=0;m<node->n_children;m++)
{
double v = node->children[m].q* (node->state.next_player == player_col?1:-1)
+ C_PUCT * node->children[m].p * sqrt_N / (node->children[m].N_effective+1);
if(v>best)
{
best = v;
index=m;
}
}
#if OMP
node->children[index].mtx.lock();
#endif
node->children[index].N_effective++;
#if OMP
node->children[index].mtx.unlock();
node->mtx.unlock();
#endif
node = node->children[index].node;
}
return node;
}
inline MCTS_Node* expand(MCTS_Node* node, double r)
{
if(is_terminal(&node->state))
{
#if OMP
node->mtx.unlock();
#endif
return node;
}
node->expanded=true;
vector<move_t*> moves;
possible_moves(&node->state, moves);
vector<edge_t> edges;
for(int i=0;i<moves.size();i++)
{
edge_t e;
e.move=moves[i];
e.value=move_heuristic_value(&node->state, e.move);
edges.push_back(e);
}
sort(edges.begin(),edges.end(), sorting_cmp);
double sum_p = 0;
double q = 1;
for(int i=0;i<moves.size();i++)
{
sum_p+=q;
q*=FACTOR;
}
q=1;
node->n_children = moves.size();
for(int e=0;e<edges.size();e++)
{
move_t* m = edges[e].move;
MCTS_Node* child_node = new_node();
node->children[e].move_id = m->move_id;
node->children[e].parent = node;
node->children[e].node = child_node;
node->children[e].N = 0;
node->children[e].N_effective = 0;
node->children[e].W = 0;
node->children[e].q = 100000;
child_node->parent = node->children+e;
child_node->expanded = false;
apply_move(&node->state, m);
clone_state(&child_node->state, &node->state);
reverse_move(&node->state, m);
node->children[e].p=q/sum_p;
q*=FACTOR;
}
for(int i=0;i<moves.size();i++)
free(moves[i]);
int n = 0;
while(r>node->children[n].p)
{
r-=node->children[n].p;
n++;
}
node->children[n].N_effective++;
#if OMP
node->mtx.unlock();
#endif
return node->children[n].node;
}
int MCTSPlayer::rollout(state_t* state, std::uniform_real_distribution<double> &t_dist, std::default_random_engine &t_gen)
{ 
while(!is_terminal(state))
{
move_t m = get_random_move(state, t_dist, t_gen);
apply_move(state, &m);
}
return get_winner(state)==player_col?1:-1;
}
inline void backup(MCTS_Node* node, int v)
{
MCTS_Edge* edge = node->parent;
while(edge!=NULL)
{
#if OMP
edge->mtx.lock();
#endif
edge->N++;
edge->W+=v;
edge->q = (1.0*edge->W)/(edge->N);
#if OMP
edge->mtx.unlock();
#endif
edge = edge->parent->parent;
}
}
void free_node(MCTS_Node* node)
{
nodes_counter--;
if(node->expanded)
for(int i=0;i<node->n_children;i++)
{
free_node(node->children[i].node);
}
free(node);
}
void MCTSPlayer::descend_from_root(int move_id)
{
MCTS_Node* new_root= NULL;
int N = root->n_children;
#if OMP
#pragma omp parallel for
#endif
for(int n=0;n<N;n++)
{
if(root->children[n].move_id==move_id)
{
new_root = root->children[n].node;
new_root->parent=NULL;
}
else
free_node(root->children[n].node);
}
free(root);
root = new_root;
nodes_counter--;
}
pair<int, double> MCTSPlayer::next_move(state_t* state, move_t* prev_move)
{
player_col = state->next_player;
adv_col = WHITE + BLACK - player_col;
auto start_time = chrono::system_clock::now();
if(root==NULL)
{
root = new_node();
clone_state(&root->state, state);
}
else
descend_from_root(prev_move->move_id);
int rollouts_count = 0;
#if OMP
mutex mtx;
#pragma omp parallel
{
int thread_id = omp_get_thread_num();
std::uniform_real_distribution<double> t_dist;
std::default_random_engine t_gen;
t_gen.seed((int)(u_dist(gen)*10000));
state_t state_c;
bool not_done;
do
{
MCTS_Node* node = select(root, player_col);
node = expand(node, t_dist(t_gen));
clone_state(&state_c, &node->state);
int v = rollout(&state_c, t_dist, t_gen);
backup(node, v);
mtx.lock();
rollouts_count++;
not_done = rollouts_count<=(max_rollouts-N_THREADS);
mtx.unlock();
} while(not_done && chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now()-start_time).count()<1000*time_budget);
}
#else
state_t state_c;
do
{
MCTS_Node* node = select(root, player_col);
node = expand(node, u_dist(gen));
clone_state(&state_c, &node->state);
int v = rollout(&state_c, u_dist, gen);
backup(node, v);
rollouts_count++;
} while(rollouts_count<= max_rollouts && chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now()-start_time).count()<1000*time_budget);
#endif
int move_id = -1;
int best = -1;
for(int m=0;m<root->n_children;m++)
{
int N = root->children[m].N;
if(N>best)
{
best = N;
move_id = root->children[m].move_id;
}
}
descend_from_root(move_id);
return pair<int, double>(move_id, rollouts_count);
}