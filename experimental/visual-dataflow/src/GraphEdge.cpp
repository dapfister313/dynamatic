//===- GraphEdge.cpp - Represents an edge in a graph ------------*- C++ -*-===//
//
// This file contains the implementation of a GraphEdge.
//
//===----------------------------------------------------------------------===//
#include "GraphEdge.h"
#include <utility>

using namespace dynamatic::experimental::visual_dataflow;

GraphEdge::GraphEdge(EdgeId id, GraphNode *src, GraphNode *dst, int inPort,
                     int outPort, std::vector<std::pair<float, float>> position)
    : id(id), src(src), dst(dst), inPort(inPort), outPort(outPort),
      position(std::move(position)) {}