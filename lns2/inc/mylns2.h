#pragma once
#include "common.h"
#include "Instance.h"
#include "BasicLNS.h"
#include "string"

// ROS includes
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/MarkerArray.h>
#include <signal.h>

enum init_destroy_heuristic { TARGET_BASED, COLLISION_BASED, RANDOM_BASED, INIT_COUNT };

class MyLns2 {
public:
    // MyLns2(int seed, vector<vector<vector<int>>> obs_map, vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss, int all_ag_num, int map_size);
    MyLns2(int seed, vector<vector<int>> obs_map, vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,int all_ag_num,int map_size);
    ~MyLns2();
    int makespan=0;
    void init_pp();
    vector<vector<pair<int,int>>> vector_path;
    vector<vector<pair<int,int>>> sipps_path;
    int calculate_sipps(vector<int> new_agents);
    int single_sipp(vector<vector<pair<int,int>>> dy_obs_path,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,
                    pair<int,int> self_start_poss,pair<int,int> self_goal_poss,int global_num_agent);

    //ROS
    ros::Publisher marker_pub;  // Add this as a member variable
    pid_t bag_recorder_pid = -1;
    bool myros = true;  // Set to true when ROS visualization should be active
    vector<vector<vector<int>>> dynamic_map_seq;

private:
    Neighbor neighbor;
    const Instance instance;
    vector<Agent> agents;
    PathTableWC path_table;
    bool updateCollidingPairs(set<pair<int, int>>& colliding_pairs,int agent_id, const Path& path);
    void ensure_ros_initialized();
    void publishMapImage(const std::vector<std::vector<int>>& map, const std::vector<int>& start_locations);
    void startBagRecording(const std::string& filename, const std::string& topic);
    void stopBagRecording();    
};