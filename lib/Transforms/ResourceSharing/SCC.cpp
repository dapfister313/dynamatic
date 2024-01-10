/*
 *     Implementation of Kosaraju's algorithm
 *     Explanatory video: https://www.youtube.com/watch?v=Qdh6-a_2MxE&t=328s
 *     Tested using: https://www.geeksforgeeks.org/problems/strongly-connected-components-kosarajus-algo/1?utm_source=geeksforgeeks&utm_medium=ml_article_practice_tab&utm_campaign=article_practice_tab
 */

#include "dynamatic/Transforms/ResourceSharing/SCC.h"

#include <vector>
#include <list>
#include <iostream>
#include <stack>
#include <algorithm>

//prints a specific array of adjacency lists to the console
void print_list(std::vector<std::list<int>>& adjacency_list, int Nodes) {
    for(int i = 0; i < Nodes; i++) {
        std::cout << i << ": ";
        for(auto item : adjacency_list[i]) {
            std::cout << item << ", ";
        }
        std::cout << "\n";
    }
}

//prints a stack to the console
void print_stack(std::stack<int> DFSstack) {
    std::cout << "Printing stack: ";
    while(!DFSstack.empty()) {
        std::cout << DFSstack.top() << " ";
        DFSstack.pop();
    }
    std::cout << "\n";
}

unsigned int getNumberOfBBs(SmallVector<experimental::ArchBB> archs) {
    unsigned int maximum = 0;
    for(auto arch_item : archs) {
      maximum = std::max(maximum, std::max(arch_item.srcBB, arch_item.dstBB));
    }
    return maximum + 1; //as we have BB0, we need to add one at the end
}

//creates an array of adjacency lists
std::vector<std::list<int>> create_adjacency_list_bbl(SmallVector<experimental::ArchBB> archs, int Nodes) {
    std::vector<std::list<int>> result(Nodes);
    for(auto arch_item : archs) {
      result[arch_item.srcBB].push_front(arch_item.dstBB);
    }
    return result;
}
std::vector<std::list<int>> create_adjacency_list_gfg(std::vector<std::vector<int>>& adj, int V) {
    std::vector<std::list<int>> result(V);
    for(unsigned long i = 0; i < adj.size(); i++) {
        for(auto item : adj[i]) {
            result[i].push_front(item);
        }
    }
    return result;
}

//calls itself to recursively traverse the whole graph to perform DFS
void firstRecursiveDFStravel(std::stack<int>& DFSstack, std::vector<bool>& node_visited, std::vector<std::list<int>>& adjacency_list, int Nodes, int current_node) {
    std::list<int> current_list = adjacency_list[current_node];
    for(auto item : current_list) {
        if(!node_visited[item]) {
            node_visited[item] = true;
            firstRecursiveDFStravel(DFSstack, node_visited, adjacency_list, Nodes, item);
        }
    }
    DFSstack.push(current_node);
}

//performs the first step of kosaraju's algorithm -> first DFS travel, recording the visit ordering of the nodes
void firstDFStravel(std::stack<int>& DFSstack, std::vector<std::list<int>>& adjacency_list, int Nodes) {
    std::vector<bool> node_visited(Nodes, false);
    //Every BB can inevitably be reached from BB0
    int current_node = 0;
    node_visited[0] = true;
    firstRecursiveDFStravel(DFSstack, node_visited, adjacency_list, Nodes, current_node);
    /*
    //This code part is only used for algorithm verification using gfg
    //The assumption here is, that not all nodes can be reached through node 0
    bool continue_it = true;
    while(continue_it) {
        continue_it = false;
        for(int i = 0; i < Nodes; i++) {
            if(!node_visited[i]) {
                node_visited[i] = true;
                firstRecursiveDFStravel(DFSstack, node_visited, adjacency_list, Nodes, i);
                continue_it = true;
                break;
            }
        }
    }
    */
}

//performs the second step of kosaraju's algorithm ->  reverse original graph
std::vector<std::list<int>> converse_graph(std::vector<std::list<int>>& adjacency_list, int Nodes) {
    std::vector<std::list<int>> result(Nodes);
    for(int i = 0; i < Nodes; i++) {
        for(auto item : adjacency_list[i]) {
            result[item].push_front(i);
        }
    }
    return result;
}

//calls itself to recursively traverse the whole graph to perform DFS
void secondRecursiveDFStravel(std::vector<std::list<int>>& transpose_graph, int Nodes, std::vector<bool>& node_visited, int current_node, std::list<int>& currSCC) {
    std::list<int> current_list = transpose_graph[current_node];
    for(auto item : current_list) {
        if(!node_visited[item]) {
            node_visited[item] = true;
            secondRecursiveDFStravel(transpose_graph, Nodes, node_visited, item, currSCC);
        }
    }
    currSCC.push_front(current_node);
}

//performs the third step of kosaraju's algorithm -> DFS travel on the converse graph, recovering SCCs
std::vector<std::list<int>> secondDFStravel(std::vector<std::list<int>> transpose_graph, std::stack<int> DFSstack, int Nodes) {
    std::vector<std::list<int>> result;
    std::vector<bool> node_visited(Nodes, false);
    while(!DFSstack.empty()) {
        int current_node = DFSstack.top();
        DFSstack.pop();
        if(node_visited[current_node]) {
            continue;  
        }
        node_visited[current_node] = true;
        std::list<int> currSCC;
        secondRecursiveDFStravel(transpose_graph, Nodes, node_visited, current_node, currSCC);
        result.push_back(currSCC);
    }
    return result;
}

//function to find the strongly connected components of a graph
std::vector<int> Kosarajus_algorithm_BBL(SmallVector<experimental::ArchBB> archs) {
    int Nodes = getNumberOfBBs(archs);
    std::vector<int> result(Nodes);
    std::vector<std::list<int>> adjacency_list = create_adjacency_list_bbl(archs, Nodes);
    std::stack<int> DFSstack;
    firstDFStravel(DFSstack, adjacency_list, Nodes);
    std::vector<std::list<int>> transpose_graph = converse_graph(adjacency_list, Nodes);
    std::vector<std::list<int>> SCC = secondDFStravel(transpose_graph, DFSstack, Nodes);
    int position = 0;
    for(auto set : SCC) {
        for(auto number : set) {
            result[number] = position;
        }
        ++position;
    }
    return result;
}