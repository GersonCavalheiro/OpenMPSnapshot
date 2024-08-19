#pragma once
#include "common.h"
#include "graph.hh"
#include "pattern.h"


void SglSolver(Graph &g, Pattern &p, uint64_t &total);
void SglVerifier(const Graph &g, uint64_t test_total);
