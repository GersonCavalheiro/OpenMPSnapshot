#pragma once

#include "TreeNode.h"

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <climits>
#include <omp.h>
#include <thread>

class MonteCarloTree {
public:
std::unique_ptr<TreeNode> root;

board root_board;

std::random_device rd;
std::default_random_engine eng;
double c_virtual_loss = 5;
static constexpr double explore_parameter = sqrt(2.0);

MonteCarloTree() : root(), root_board(), eng(rd()) {}

private:
TreeNode* UCB (TreeNode* n)  {

if(n->child_size == 0) return nullptr;

constexpr double eps = 1e-3;
double max_score = INT_MIN;
std::size_t same_score[100]{};
std::size_t idx = 0;

for (std::size_t i = 0; i < n->child_size; ++i) {
TreeNode* childNode = n->child.get() + i;

int child_win_count = childNode->win_count.load();
int child_total_count = childNode->total_count.load();

double virtual_loss = c_virtual_loss * n->virtual_loss.load();

const double exploit { child_win_count / (double)( child_total_count + 1.0) };
const double explore { sqrt( log( n->total_count.load() ) / (double)( child_total_count + 1.0) ) };


const double score { exploit + explore_parameter * explore - virtual_loss };

if ( (score <= (max_score + eps) ) && (score >= (max_score - eps) ) ) {
same_score[idx] = i;
idx++;
}
else if (score > max_score) {
max_score = score;
same_score[0] = i;
idx = 1;
}
}

std::shuffle(std::begin(same_score), std::begin(same_score) + idx, eng);
std::size_t best_idx = same_score[0];

n->virtual_loss++;

return (n->child.get() + best_idx); 
}

void select(board &b, std::vector<TreeNode*> &path) {

TreeNode* current { root.get() };

path.push_back(current);

while (current->child != nullptr && current->child_size != 0) {
current = UCB(current);
path.push_back(current);

b.move(current->move.prev, current->move.next, current->color);
}
}



bool RandomRollout(board &b, PIECE color) {
std::vector<Pair> mvs { b.get_available_move(color) };

if (!mvs.empty()) {
std::shuffle(mvs.begin(), mvs.end(), eng);
b.move(mvs[0].prev, mvs[0].next, color);
} else {
return false;
}
return true;

}

WIN_STATE simulate(board b) {

std::size_t count_step = 0;

constexpr std::size_t limit_step = 500;
while (true) {
count_step++;
if (count_step > limit_step) {
return b.compare_piece();
}

const PIECE& color { b.take_turn() };
bool succ;
succ = RandomRollout(b, color);

if ( !succ ) {
if (color == BLACK)
return WHITE_WIN;
else
return BLACK_WIN;
}


}
}

void backpropogate(const WIN_STATE &result, std::vector<TreeNode*> &path) {
for (auto &node : path) {
node->addresult(result);
}
}

public:

void tree_policy() {
board b {root_board};
TreeNode *current;
std::vector<TreeNode*> path;

select(b, path);

TreeNode &leaf_node = *(path.back());


if (leaf_node.child_size==0 && leaf_node.total_count.load() > 0){

leaf_node.expand(b);

if (leaf_node.child_size != 0) {
current = UCB(&leaf_node);
path.push_back(current);
b.move(current->move.prev, current->move.next, current->color);
}
else {
const WIN_STATE result = ( (leaf_node.color==WHITE) ? WHITE_WIN : BLACK_WIN);
backpropogate(result, path);
return;
}
}

const WIN_STATE result { simulate(b) };

backpropogate(result, path);
}

void leafParallel(board b, std::vector<TreeNode*> &path) {
auto result = simulate(b);
backpropogate(result, path);
}


void leafPthread(board b, std::vector<TreeNode*> &path, EnvParameter env) {
std::thread workers[ env.thread_num ];
for(int i=1; i < env.thread_num; i++) {
workers[i] = std::thread(&MonteCarloTree::leafParallel, this, b, std::ref(path));
}

leafParallel(b, path);
for(int i=1; i < env.thread_num; i++) {
workers[i].join();
}
}

void leafOMP(board b, std::vector<TreeNode*> &path, EnvParameter env) {
omp_set_num_threads(env.thread_num);

#pragma omp parallel for
for (int i=0; i < env.thread_num; ++i){

auto result = simulate(b);

#pragma omp critical
backpropogate(result, path);
}
}
void parallelLeaf_tree_policy(const EnvParameter &env) {
board b {root_board};
TreeNode *current;
std::vector<TreeNode*> path;

select(b, path);

TreeNode &leaf_node = *(path.back());


if (leaf_node.child_size==0 && leaf_node.total_count.load() > 0){

leaf_node.expand(b);

if (leaf_node.child_size != 0) {
current = UCB(&leaf_node);
path.push_back(current);
b.move(current->move.prev, current->move.next, current->color);
}
else {
const WIN_STATE result = ( (leaf_node.color==WHITE) ? WHITE_WIN : BLACK_WIN);
backpropogate(result, path);
return;
}
}

if (root->color == BLACK) {
if (env.black_method[0] == 'o') {	
leafOMP(b, path, env);
}
else if (env.black_method[0] == 'p') {
leafPthread(b, path, env);
}
} else {
if (env.white_method[0] == 'o') {	
leafOMP(b, path, env);
}
else if (env.white_method[0] == 'p') {
leafPthread(b, path, env);
}
}

}


void parallelTree_tree_policy() {
board b {root_board};
TreeNode *current;
std::vector<TreeNode*> path;

select(b, path);

TreeNode &leaf_node = *(path.back());


if (leaf_node.child_size==0 && leaf_node.total_count.load() > 0){

leaf_node.expandLock(b);

if (leaf_node.child_size != 0) {
current = UCB(&leaf_node);
path.push_back(current);
b.move(current->move.prev, current->move.next, current->color);
}
else {
const WIN_STATE result = ( (leaf_node.color==WHITE) ? WHITE_WIN : BLACK_WIN);
backpropogate(result, path);
return;
}
}

const WIN_STATE result { simulate(b) };

backpropogate(result, path);
}


void reset(board &b) {
root_board = b;
root = { std::make_unique<TreeNode>() };
root->color = root_board.take_turn();
root->move = {-1, -1};
root->total_count.store(0);
root->win_count.store(0);
root->child = nullptr;
root->child_size = 0;
root->expand(b);
}

};