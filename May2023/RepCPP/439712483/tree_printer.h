#pragma once

#include "knode.h"

#include <iostream>

template class KNode<data_type>;


std::ostream &operator<<(std::ostream &os, const KNode<data_type> &node);


std::ostream &print_node_values(std::ostream &os, const KNode<data_type> &node);
