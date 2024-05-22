#pragma once
#include "common.h"
#include "Instance.h"
#include "BasicLNS.h"
#include "string"

enum init_destroy_heuristic { TARGET_BASED, COLLISION_BASED, RANDOM_BASED, INIT_COUNT};

class MyLns2 {
public:
    MyLns2(int seed,vector<vector<int>> obs_map,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss, int all_ag_num, int map_size);
    int num_of_colliding_pairs=0;
    int makespan=0;
    void init_pp();
    vector<vector<pair<int,int>>> vector_path;
    vector<vector<pair<int,int>>> sipps_path;
    vector<vector<pair<int,int>>> add_sipps_path;
    vector<vector<pair<int,int>>> replan_sipps_path;
    vector<int> used_shuffled_agents;
    int old_coll_pair_num=0;
    PathTableWC path_table;
    bool select_and_sipps(bool if_update,bool first_flag,vector<vector<pair<int,int>>> new_path,set<pair<int, int>> new_collsion_pair,int num_local_ag);
    void extract_path();
    int add_sipps(vector<vector<pair<int,int>>> temp_path,vector<int> selected_ag, const vector<pair<int,int>>& vector_start_locations);
    void replan_sipps(vector<vector<pair<int,int>>> temp_path,vector<int> selected_ag);
    Neighbor add_neighbor;
    Neighbor neighbor;
    Neighbor replan_neighbor;

private:
    const Instance instance;
    vector<Agent> agents;
    vector<set<int>> collision_graph;
    vector<int> goal_table;
    double decay_factor = -1;
    void runPP();
    init_destroy_heuristic init_destroy_strategy = COLLISION_BASED;
    double reaction_factor = -1;
    vector<double> destroy_weights;
    int randomWalk(int agent_id);
    int selected_neighbor;
    int neighbor_size;
    void chooseDestroyHeuristicbyALNS();
    bool generateNeighborByCollisionGraph();
    bool generateNeighborByTarget();
    bool generateNeighborRandomly();
    void rouletteWheel();
    bool add_updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path,const PathTableWC& add_path_table,const vector<Agent>& add_agents);
    bool updateCollidingPairs(set<pair<int, int>>& colliding_pairs,int agent_id, const Path& path);
    static unordered_map<int, set<int>>& findConnectedComponent(const vector<set<int>>& graph, int vertex,
                                                                unordered_map<int, set<int>>& sub_graph);
};
