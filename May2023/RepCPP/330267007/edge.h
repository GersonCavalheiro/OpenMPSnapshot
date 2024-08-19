#pragma once


template <typename V, typename E, typename W>
class Edge {
public:
Edge(V source, E label, W weight, V target);

V source;
E label;
W weight;
V target;
};

template <typename V, typename E, typename W>
Edge<V,E,W>::Edge(V source, E label, W weight, V target) {
this->source = source;
this->label = label;
this->weight = weight;
this->target = target;
}

