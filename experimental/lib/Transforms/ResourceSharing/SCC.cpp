/*
 *     [info] This file contains an algorithm that takes a graph with n nodes
 *            and returns the Strongly connected components in this graph
 *     [info] The algorithm works well with low edge to node ratio
 *            If this is not the case one might consider usng an other algorithm
 *     Implementation of Kosaraju's algorithm
 *     Explanatory video: https://www.youtube.com/watch?v=Qdh6-a_2MxE&t=328s
 *     Tested using: https://www.geeksforgeeks.org/problems/strongly-connected-components-kosarajus-algo/1?utm_source=geeksforgeeks&utm_medium=ml_article_practice_tab&utm_campaign=article_practice_tab
 */

#include "experimental/Transforms/ResourceSharing/SCC.h"

#include <vector>
#include <list>
#include <iostream>
#include <stack>
#include <algorithm>

using namespace dynamatic::experimental;
using namespace dynamatic::handshake;
using namespace llvm;
using namespace dynamatic;


namespace dynamatic {
namespace experimental {
namespace sharing {


/*
 * Dumps content of vector of lists to the console
 */
void print_list(std::vector<std::list<int>>& adjacency_list) {
    int Nodes = adjacency_list.size();
    for(int i = 0; i < Nodes; i++) {
        llvm::errs() << i << ": ";
        for(auto item : adjacency_list[i]) {
            llvm::errs() << item << ", ";
        }
        llvm::errs() << "\n";
    }
    return;
}

/*
 * Dumps content of stack to the console
 */
void print_stack(std::stack<int> DFSstack) {
    std::cout << "Printing stack: ";
    while(!DFSstack.empty()) {
        std::cout << DFSstack.top() << " ";
        DFSstack.pop();
    }
    std::cout << "\n";
    return;
}

/*
 * Gets the number of Basic Blocks in the IR
 */
unsigned int getNumberOfBBs(SmallVector<ArchBB> archs) {
    unsigned int maximum = 0;
    for(auto arch_item : archs) {
      maximum = std::max(maximum, std::max(arch_item.srcBB, arch_item.dstBB));
    }
    //as we have BB0, we need to add one at the end
    return maximum + 1;
}

/*
 * Input: Container with source/destination Basic Blocks of each Edge between Basic Blocks
 * Output: vector of lists, vector[n] stores all destination Basic Blocks, where the
 *         source Basic Block is the n.th Basic Block
 * Example: (In: (0,1),(1,2),(2,2),(2,3) )
             Out:  0: 1
                   1: 2
                   2: 2,3
                   3: -
 */
std::vector<std::list<int>> create_adjacency_list_bbl(SmallVector<ArchBB> archs, int Nodes) {
    std::vector<std::list<int>> result(Nodes);
    for(auto arch_item : archs) {
      result[arch_item.srcBB].push_front(arch_item.dstBB);
    }
    return result;
}

/*
 * This function simply converts a vector to a list
 * Use: This function is a version of create_adjacency_list_bbl to test
 *      the algorithm (see top of this file) using "geeks for geeks"
 */
std::vector<std::list<int>> create_adjacency_list_gfg(std::vector<std::vector<int>>& adj, int V) {
    std::vector<std::list<int>> result(V);
    for(unsigned long i = 0; i < adj.size(); i++) {
        for(auto item : adj[i]) {
            result[i].push_front(item);
        }
    }
    return result;
}

/*
 * Type: Recursive depth first search travel
 * Use:  Creates a stack with the last finishing nodes at the top
 * Remarks: first node chosen by function firstDFStravel
 * Example:    1 - 2 - 4   We are starting at Node 1, we get to Node 2
 *                 |       then 4, where we can no longer travel, we push
 *                 3       Node 4 on the stack, return to 2, travel to 3,
 *                         where we are again stuck, we push 3.
 *                         We return to 2 and as we already visited 1,3,4
 *                         we are stuck again and push 2, we finally return
 *                         to Node 1 and push 1 to the stack
 *                         Output: 4,3,2,1
 */
void firstRecursiveDFStravel(std::stack<int>& DFSstack, std::vector<bool>& node_visited, std::vector<std::list<int>>& adjacency_list, int Nodes, int current_node) {
    std::list<int> current_list = adjacency_list[current_node];
    for(auto item : current_list) {
        if(!node_visited[item]) {
            node_visited[item] = true;
            firstRecursiveDFStravel(DFSstack, node_visited, adjacency_list, Nodes, item);
        }
    }
    DFSstack.push(current_node);
    return;
}

/*
 * First we choose the starting node for our algorithm. As we are working with an IR, all nodes are
 * reachable from node 0, aka start.
 * Then we are just calling function firstRecursiveDFStravel
 */
void firstDFStravel(std::stack<int>& DFSstack, std::vector<std::list<int>>& adjacency_list, int Nodes) {
    std::vector<bool> node_visited(Nodes, false);
    //Every BB can inevitably be reached from BB0
    int current_node = 0;
    //As we start with node 0, we mark it as visited
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
    return;
}

/*
 * This function takes a directed graph and inverts the edges
 * Example: The edge 1->2 gets replaced with 2->1
 */
std::vector<std::list<int>> converse_graph(std::vector<std::list<int>>& adjacency_list, int Nodes) {
    std::vector<std::list<int>> result(Nodes);
    for(int i = 0; i < Nodes; i++) {
        for(auto item : adjacency_list[i]) {
            result[item].push_front(i);
        }
    }
    return result;
}

/*
 * This function does the same as function firstRecursiveDFStravel with the key difference that
 * the wanted result here is a list instead of a stack
 */
void secondRecursiveDFStravel(std::vector<std::list<int>>& transpose_graph, int Nodes, std::vector<bool>& node_visited, int current_node, std::list<int>& currSCC) {
    std::list<int> current_list = transpose_graph[current_node];
    for(auto item : current_list) {
        if(!node_visited[item]) {
            node_visited[item] = true;
            secondRecursiveDFStravel(transpose_graph, Nodes, node_visited, item, currSCC);
        }
    }
    currSCC.push_front(current_node);
    return;
}

/*
 * using the stack built in function firstDFStravel, we do a second DFS travel and store the first seen Nodes in a list
 * which we finally push onto a other container. This is done till all nodes are marked visited.
 */
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

/*
 * Input: Container with source/destination Basic Blocks of each Edge between Basic Blocks
 * Output: vector of the size of nodes, vector[n] returns the SCC id the n.th node belongs to.
 */
std::vector<int> Kosarajus_algorithm_BBL(SmallVector<ArchBB> archs) {
    int Nodes = getNumberOfBBs(archs);
    std::vector<int> result(Nodes);
    std::vector<std::list<int>> adjacency_list = create_adjacency_list_bbl(archs, Nodes);
    std::stack<int> DFSstack;
    firstDFStravel(DFSstack, adjacency_list, Nodes);
    std::vector<std::list<int>> transpose_graph = converse_graph(adjacency_list, Nodes);
    std::vector<std::list<int>> SCC = secondDFStravel(transpose_graph, DFSstack, Nodes);
    int position = 0;
    //for(unsigned long i = 0; i < SCC.size(); i++) {
        for(auto component : SCC) {
        bool is_loop_nest = true;
        if(component.size() == 1) {
            //may be not a CFG loop nest
            is_loop_nest = false;
            unsigned long BB = *component.begin();
            for(unsigned long item : adjacency_list[BB]) {
                if(item == BB) {
                    //CFG loop nest
                    is_loop_nest = true;
                }
            }
        }
        for(auto number : component) {
            if(is_loop_nest) {
                result[number] = position;
            } else {
                result[number] = -1;
            }
        }
        ++position;
    }
    return result;
}

void recursive_list_creator_opl(mlir::Operation* currOp, std::vector<std::list<int>>& adjacency_list, std::map<mlir::Operation*, int>& conversions, int* map_size, std::set<mlir::Operation*>& node_visited, std::map<Operation *, unsigned int>& topological_sort, int *sorting_idx, std::stack<int>& DFSstack) {
    node_visited.insert(currOp);
    int op_idx = conversions.find(currOp)->second;
    for (auto &u : currOp->getResults().getUses()) {
        //get child operation
        Operation *child_op = u.getOwner();

        //update adjacency list
        int child_idx = -1;
        auto cit = conversions.find(child_op);
        if(cit == conversions.end()) {
            //operation not yet present
            conversions[child_op] = *map_size;
            child_idx = *map_size;
            ++*map_size;
            adjacency_list.push_back(std::list<int>());
        } else {
            child_idx = cit->second;
        }
        adjacency_list[op_idx].push_back(child_idx);

        //traverse child operation if not yet done
        auto it = node_visited.find(child_op);
        if(it == node_visited.end()) {
           //not visited yet
           recursive_list_creator_opl(child_op, adjacency_list, conversions, map_size, node_visited, topological_sort, sorting_idx, DFSstack);
        }
    }
    topological_sort[currOp] = *sorting_idx;
    DFSstack.push(op_idx);
    *sorting_idx++;
    return;
}

void create_adjacency_list_opl(mlir::Operation* startOp, std::vector<std::list<int>>& adjacency_list, std::map<mlir::Operation*, int>& conversions, std::map<Operation *, unsigned int>& topological_sort, std::stack<int>& DFSstack) {
    std::set<mlir::Operation*> node_visited;
    int position = 0;
    int sorting_idx = 0;
    recursive_list_creator_opl(startOp, adjacency_list, conversions, &position, node_visited, topological_sort, &sorting_idx, DFSstack);
    return;
}

void get_noncyclic_operations(std::vector<std::list<int>>& adjacency_list, std::vector<std::list<int>>& SCC, std::map<mlir::Operation*, int>& conversions, std::set<mlir::Operation*>& result) {
    //check which indexes are noncyclic
    std::set<int> intermediate;
    for(auto list : SCC) {
        if(list.size() == 1) {
            //may be noncyclic
            int value = *list.begin();
            for(auto item : adjacency_list[value]) {
                if(item == value) {
                    continue;
                }
            }
            intermediate.insert(value);
        }
    }
    //convert indexes to actual operations
    for(auto it : conversions) {
        if(intermediate.find(it.second) != intermediate.end()) {
            result.insert(it.first);
        }
    }
    return;
}

void Kosarajus_algorithm_OPL(mlir::Operation* startOp, std::set<mlir::Operation*>& result, std::map<Operation *, unsigned int>& topological_sort) {
    std::vector<std::list<int>> adjacency_list;
    std::map<mlir::Operation*, int> conversions;
    std::stack<int> DFSstack;
    create_adjacency_list_opl(startOp, adjacency_list, conversions, topological_sort, DFSstack);
    int Nodes = adjacency_list.size();
    std::vector<std::list<int>> transpose_graph = converse_graph(adjacency_list, Nodes);
    std::vector<std::list<int>> SCC = secondDFStravel(transpose_graph, DFSstack, Nodes);
    get_noncyclic_operations(adjacency_list, SCC, conversions, result);
    return;
}

} // namespace sharing
} // namespace experimental
} // namespace dynamatic
