#include "mylns2.h"
#include <iostream>
#include <random>
#include "common.h"
#include <utility>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/MarkerArray.h>
#include <cstdlib>
#include <unistd.h>

/*
MyLns2::MyLns2(int seed, vector<vector<vector<int>>> obs_map, vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss, int all_ag_num, int map_size)
    : instance(obs_map, start_poss, goal_poss, all_ag_num, map_size),
      path_table(map_size * map_size, all_ag_num)
*/
      
MyLns2::MyLns2(int seed, vector<vector<int>> obs_map,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,int all_ag_num,int map_size):
    instance(obs_map,start_poss,goal_poss,all_ag_num,map_size),path_table(map_size*map_size, all_ag_num),bag_recorder_pid(-1)  // Initialize bag recorder PID
{
    if (myros) {
        ensure_ros_initialized();
    
        if (ros::isInitialized()) {
            static ros::NodeHandle nh;
            std::string env_suffix = std::to_string(seed);
            std::string marker_topic = "/sipp/iteration_markers_" + env_suffix;
    
            marker_pub = nh.advertise<visualization_msgs::MarkerArray>(marker_topic, 1, true);
    
            // ✅ Only publish the first frame of dynamic map
            publishMapImage(obs_map, instance.start_locations);
    
            // Start recording
            std::string bag_filename = "/lns2rl/Ros/Bags/sipp_vis_" + env_suffix + ".bag";
            startBagRecording(bag_filename, marker_topic);
        }
    }

    srand(seed);
    agents.reserve(all_ag_num);
    for (int i = 0; i < all_ag_num; i++)
        agents.emplace_back(instance, i, instance.start_locations, instance.goal_locations);
}

MyLns2::~MyLns2() {
    if (myros) stopBagRecording();  // Clean up on destruction
}

void MyLns2::ensure_ros_initialized() {
    if (!ros::isInitialized()) {
        int argc = 0;
        char **argv = nullptr;
        ros::init(argc, argv, "lns2_pybind_node", ros::init_options::AnonymousName);
        std::cout << "[ROS] ros::init() called with anonymous name\n";
    }
}

void MyLns2::publishMapImage(const std::vector<std::vector<int>>& map,
    const std::vector<int>& start_locations)
{
    visualization_msgs::MarkerArray marker_array;
    int width = map[0].size();
    int height = map.size();
    int id = 0;

    // (1) Base map cubes (same as before)...

    // (2) Count start location overlaps
    std::unordered_map<int, int> start_counts;
    for (int loc : start_locations)
    start_counts[loc]++;

    for (const auto& [loc, count] : start_counts) {
    int x = loc % width;
    int y = loc / width;

    visualization_msgs::Marker sphere;
    sphere.header.frame_id = "map";
    sphere.header.stamp = ros::Time::now();
    sphere.ns = "agent_starts";
    sphere.id = id++;
    sphere.type = visualization_msgs::Marker::SPHERE;
    sphere.action = visualization_msgs::Marker::ADD;
    sphere.pose.position.x = x + 0.5;
    sphere.pose.position.y = y + 0.5;
    sphere.pose.position.z = 0.2;
    sphere.scale.x = 0.4;
    sphere.scale.y = 0.4;
    sphere.scale.z = 0.4;
    sphere.color.a = 1.0;

    if (count > 1) {
    sphere.color.r = 0.5;  // purple for overlapping
    sphere.color.g = 0.0;
    sphere.color.b = 0.5;
    } else {
    sphere.color.r = 0.0;  // green for unique
    sphere.color.g = 1.0;
    sphere.color.b = 0.0;
    }

    marker_array.markers.push_back(sphere);
    }

    marker_pub.publish(marker_array);
    ros::spinOnce();
    ros::Duration(0.05).sleep();
}

void MyLns2::startBagRecording(const std::string& filename, const std::string& topic) {
    pid_t pid = fork();
    if (pid == 0) {
        // Child process: setup safe ROS-only LD_LIBRARY_PATH
        setenv("LD_LIBRARY_PATH", "/opt/ros/noetic/lib", 1);

        execlp("rosbag", "rosbag", "record",
            "--split",
            "--size=4096",           // Correct argument: not "--max-bag-size"
            "-O", filename.c_str(),
            topic.c_str(),
            (char*) nullptr);

        std::cerr << "Failed to start rosbag record" << std::endl;
        std::_Exit(1);
    } else if (pid > 0) {
        bag_recorder_pid = pid;
        std::cout << "[ROS] rosbag record started (pid: " << bag_recorder_pid << ")" << std::endl;
    } else {
        std::cerr << "[ROS] Failed to fork rosbag record" << std::endl;
    }
}

void MyLns2::stopBagRecording() {
    if (bag_recorder_pid > 0) {
        kill(bag_recorder_pid, SIGINT);  // sends Ctrl+C
        std::cout << "[ROS] rosbag record stopped" << std::endl;
    }
}

void MyLns2::init_pp()
{
    for (int i = 0; i < (int)agents.size(); i++) {
        neighbor.agents.push_back(i);
        auto* sipp_solver = dynamic_cast<SIPP*>(agents[i].path_planner);
        if (sipp_solver) {
            sipp_solver->marker_pub = marker_pub;
        }
    }
    neighbor.agents.reserve(agents.size());
    for (int i = 0; i < (int)agents.size(); i++)
        neighbor.agents.push_back(i);
    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    for (auto id : neighbor.agents)
    {
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        // Add this before the assertion
    if (agents[id].path.empty()) {
        std::cout << "ERROR: Agent " << id << " has empty path after planning" << std::endl;
        std::cout << "  Start location: " << agents[id].path_planner->start_location << std::endl;
        std::cout << "  Goal location: " << agents[id].path_planner->goal_location << std::endl;
    } else if (agents[id].path.back().location != agents[id].path_planner->goal_location) {
        std::cout << "ERROR: Agent " << id << " final location mismatch" << std::endl;
        std::cout << "  Path back location: " << agents[id].path.back().location << std::endl;
        std::cout << "  Goal location: " << agents[id].path_planner->goal_location << std::endl;
    }  // no hard obstacle, thus must find path
        path_table.insertPath(agents[id].id, agents[id].path);
    }
    vector_path.reserve(instance.num_of_agents);
    for (const auto &agent : agents)
    {
        vector<pair<int,int>> single_path;
        for (const auto &state : agent.path)
            single_path.push_back(instance.getCoordinate(state.location));
        vector_path.push_back(single_path);
    }
}


bool MyLns2::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path)
{
    bool succ = false;
    if (path.size() < 2)
        return succ;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
                succ = true;
                colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));// emplace: insert new element into set
            }
        }
        if (from != to && path_table.table[to].size() >= t && path_table.table[from].size() > t) // edge conflicts(swapping conflicts)
        {
            for (auto a1 : path_table.table[to][t - 1])
            {
                for (auto a2: path_table.table[from][t])
                {
                    if (a1 == a2)
                    {
                        succ = true;
                        colliding_pairs.emplace(min(agent_id, a1), max(agent_id, a1));
                        break;
                    }
                }
            }
        }
        if (!path_table.goals.empty() && path_table.goals[to] < t) // target conflicts, already has agent in its goal, so the new agent can not tarverse it
        { // this agent traverses the target of another agent
            for (auto id : path_table.table[to][path_table.goals[to]]) // look at all agents at the goal time
            {
                if (agents[id].path.back().location == to) // if agent id's goal is to, then this is the agent we want
                {
                    succ = true;
                    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
                    break;
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < path_table.table[goal].size(); t++)
    {
        for (auto id : path_table.table[goal][t])
        {
            succ = true;
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
    return succ;
}

int MyLns2::calculate_sipps(vector<int> new_agents)
{
    neighbor.colliding_pairs.clear();
    neighbor.agents=new_agents;
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
        path_table.deletePath(neighbor.agents[i]);
    makespan=path_table.makespan;
    auto p = neighbor.agents.begin();
    sipps_path.clear();
    sipps_path.reserve(new_agents.size());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    while (p != neighbor.agents.end())
    {
        int id = *p;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        
        if (agents[id].path.empty()) {
            std::cout << "Warning: No path found for agent " << id << ". Using fallback strategy." << std::endl;
            
            // Fallback strategy options:
            // 1. Keep the agent at its start position
            Path fallback_path;
            fallback_path.push_back(PathEntry(agents[id].path_planner->start_location));
            agents[id].path = fallback_path;
            
            // 2. Or try alternative pathfinding method
            // agents[id].path = findAlternativePath(...);
        }
        
        // Now check that the path ends at the goal (or use an alternative goal if needed)
        if (agents[id].path.back().location != agents[id].path_planner->goal_location) {
            std::cout << "Warning: Agent " << id << " couldn't reach goal. Path ends at " 
                      << agents[id].path.back().location << " instead of " 
                      << agents[id].path_planner->goal_location << std::endl;
        }
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path);
        vector<pair<int,int>> single_path;
        for (const auto &state : agents[id].path)
        {   single_path.push_back(instance.getCoordinate(state.location));
        }
        sipps_path.push_back(single_path);
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
    for (auto id : neighbor.agents)
    {
        if (agents[id].path.size()==1)
        {
            int to = agents[id].path[0].location;
            int t=0;
            for (auto & ag_list : path_table.table[to])
            {
                if (t!=0 && !ag_list.empty())
                {
                    for (int another_id: ag_list)
                        neighbor.colliding_pairs.emplace(min(another_id, id), max(another_id, id));// emplace: insert new element into set
                }
                t++;
            }
        }
    }

    return (int)neighbor.colliding_pairs.size();
}

int MyLns2::single_sipp(vector<vector<pair<int,int>>> dy_obs_path,vector<pair<int,int>> start_poss, vector<pair<int,int>> goal_poss,
                                  pair<int,int> self_start_poss,pair<int,int> self_goal_poss,int global_num_agent)
{
    vector<Agent> add_agents;
    PathTableWC add_path_table(instance.map_size,global_num_agent);
    add_agents.reserve(global_num_agent); //vector.reserve: adjust capacity
    vector<int> start_locations;
    vector<int> goal_locations;
    start_locations.reserve(global_num_agent);
    goal_locations.reserve(global_num_agent);
    for(int id=0;id<global_num_agent-1;id++)
    {
        start_locations.push_back(instance.linearizeCoordinate(start_poss[id].first, start_poss[id].second));
        goal_locations.push_back(instance.linearizeCoordinate(goal_poss[id].first, goal_poss[id].second));}
    start_locations.push_back(instance.linearizeCoordinate(self_start_poss.first, self_start_poss.second));
    goal_locations.push_back(instance.linearizeCoordinate(self_goal_poss.first, self_goal_poss.second));
    for(int id=0;id<global_num_agent-1;id++)
    {
        add_agents.emplace_back(instance, id,start_locations,goal_locations);
        add_agents[id].path.resize(dy_obs_path[id].size());
        for (int t=0;t<(int)dy_obs_path[id].size();t++)
        {
            add_agents[id].path[t].location=instance.linearizeCoordinate(dy_obs_path[id][t].first, dy_obs_path[id][t].second);
        }
        add_path_table.insertPath(id, add_agents[id].path);
    }
    add_agents.emplace_back(instance, global_num_agent-1,start_locations,goal_locations);
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &add_path_table);
    add_agents[global_num_agent-1].path = add_agents[global_num_agent-1].path_planner->findPath(constraint_table);
    assert(!add_agents[global_num_agent-1].path.empty() && add_agents[global_num_agent-1].path.back().location == add_agents[global_num_agent-1].path_planner->goal_location);
    int path_ln=(int)add_agents[global_num_agent-1].path.size();
    return path_ln;

}







