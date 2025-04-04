#include "SIPP.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
namespace fs = std::filesystem;

#include <visualization_msgs/MarkerArray.h>
#include <ros/console.h>


void SIPP::updatePath(const LLNode* goal, vector<PathEntry> &path)
{
    num_collisions = goal->num_of_conflicts;
	path.resize(goal->timestep + 1);
	// num_of_conflicts = goal->num_of_conflicts;

	const auto* curr = goal;
	while (curr->parent != nullptr) // non-root node
	{
		const auto* prev = curr->parent;
		int t = prev->timestep + 1;
		while (t < curr->timestep)
		{
			path[t].location = prev->location; // wait at prev location
			t++;
		}
		path[curr->timestep].location = curr->location; // move to curr location
		curr = prev;
	}
	assert(curr->timestep == 0);
	path[0].location = curr->location;
}

// find path by A*
// Returns a path that minimizes the collisions with the paths in the path table, breaking ties by the length
Path SIPP::findPath(const ConstraintTable& constraint_table)
{
    reset();
    ReservationTable reservation_table(constraint_table, goal_location);
    Path path;
    Interval interval = reservation_table.get_first_safe_interval(start_location);
    if (get<0>(interval) > 0)
        return path;

    auto holding_time = constraint_table.getHoldingTime(constraint_table.length_min);
    auto last_target_collision_time = constraint_table.getLastCollisionTimestep(goal_location);
    auto h = max(max(my_heuristic[start_location], holding_time), last_target_collision_time + 1);

    auto start = new SIPPNode(start_location, 0, h, nullptr, 0, get<1>(interval), get<1>(interval),
                              get<2>(interval), get<2>(interval));
    pushNodeToFocal(start);

    int iteration = 0; // iteration counter for visualization
    const int MAX_ITERATIONS = 40000;
    while (!focal_list.empty())  // algorithm 1
    {   
        /*
        // Check if we've exceeded max iterations
        if (iteration >= MAX_ITERATIONS) {
            std::cout << "SIPP::findPath - FAILED: No path found from " 
                    << start_location << " to " << goal_location 
                    << " after " << MAX_ITERATIONS << " iterations" << std::endl;
            break; // Exit the search loop
        }
        */

        

        auto* curr = focal_list.top();  // top: best node in th heap
        focal_list.pop();  // remove top node
        curr->in_openlist = false;
        num_expanded++; // number of checked node
        assert(curr->location >= 0);
        // check if the popped node is a goal
        if (curr->is_goal)
        {
            updatePath(curr, path);
            break;
        }
        else if (curr->location == goal_location && // arrive at the goal location
                 !curr->wait_at_goal && // not wait at the goal location
                 curr->timestep >= holding_time) // the agent can hold the goal location afterward
        {
            int future_collisions = constraint_table.getFutureNumOfCollisions(curr->location, curr->timestep);
            if (future_collisions == 0)
            {
                updatePath(curr, path);
                break;
            }
            // generate a goal node
            auto goal = new SIPPNode(*curr);
            goal->is_goal = true;
            goal->h_val = 0;
            goal->num_of_conflicts += future_collisions;
            // try to retrieve it from the hash table
            if (dominanceCheck(goal))  // algorithm 3
                pushNodeToFocal(goal);
            else
                delete goal;
        }

        for (int next_location : instance.getNeighbors(curr->location)) // move to neighboring locations
        {
            for (auto & i : reservation_table.get_safe_intervals(
                    curr->location, next_location, curr->timestep + 1, curr->high_expansion + 1)) // algorithm 2
            {
                int next_high_generation, next_timestep, next_high_expansion;
                bool next_v_collision, next_e_collision;
                tie(next_high_generation, next_timestep, next_high_expansion, next_v_collision, next_e_collision) = i;
                if (next_timestep + my_heuristic[next_location] > constraint_table.length_max)  //<upper_bound, low, high,  vertex collision, edge collision>
                    break;
                auto next_collisions = curr->num_of_conflicts +
                                    // (int)curr->collision_v * max(next_timestep - curr->timestep - 1, 0) + // wait time
                                      (int)next_v_collision + (int)next_e_collision;
                auto next_h_val = max(my_heuristic[next_location], (next_collisions > 0?
                    holding_time : curr->getFVal()) - next_timestep); // path max
                // generate (maybe temporary) node
                auto next = new SIPPNode(next_location, next_timestep, next_h_val, curr, next_timestep,
                                         next_high_generation, next_high_expansion, next_v_collision, next_collisions);
                // try to retrieve it from the hash table
                if (dominanceCheck(next))
                    pushNodeToFocal(next);
                else
                    delete next;
            }
        }  // end for loop that generates successors
        // wait at the current location
        if (curr->high_expansion == curr->high_generation and
            reservation_table.find_safe_interval(interval, curr->location, curr->high_expansion) and
                get<0>(interval) + curr->h_val <= reservation_table.constraint_table.length_max) // up bound==t_max ,has net interval start with t_max, not exceed length limitation
        {
            auto next_timestep = get<0>(interval);  //==curr->high_expansion
            auto next_h_val = max(my_heuristic[curr->location], (get<2>(interval) ? holding_time : curr->getFVal()) - next_timestep);
            auto next_collisions = curr->num_of_conflicts +
                    // (int)curr->collision_v * max(next_timestep - curr->timestep - 1, 0) +
		    (int)get<2>(interval);
            auto next = new SIPPNode(curr->location, next_timestep, next_h_val, curr, next_timestep,
                                     get<1>(interval), get<1>(interval), get<2>(interval),
                                     next_collisions);
            next->wait_at_goal = (curr->location == goal_location);
            if (dominanceCheck(next))
                pushNodeToFocal(next);
            else
                delete next;
        }
        if (myros)
        {
            if (iteration % 5 == 0) {
                std::vector<PathEntry> partial_path;
                const auto* trace = curr;
                while (trace) {
                    PathEntry entry;
                    entry.location = trace->location;
                    partial_path.push_back(entry);
                    trace = static_cast<const SIPPNode*>(trace->parent);
                }
                std::reverse(partial_path.begin(), partial_path.end());
                rosVisualization(iteration, partial_path); // visualize partial path
            }
        }
    }  // end while loop

    /*
    // At the end of the method, before returning:
    if (path.empty()) {
        std::cout << "SIPP::findPath - FAILED: No path found from " << start_location 
                  << " to " << goal_location << std::endl;
    } else {
        std::cout << "SIPP::findPath - SUCCESS: Path found with " 
                  << path.size() << " steps" << std::endl;
        std::cout << "  Final location: " << path.back().location << std::endl;
    }
    */

    //if (path.empty())
    //{
    //    printSearchTree();
    //}
    releaseNodes();
    return path;
}

inline void SIPP::pushNodeToFocal(SIPPNode* node)
{
    num_generated++;
    allNodes_table[node].push_back(node);
    node->in_openlist = true;
    node->focal_handle = focal_list.push(node); // we only use focal list; no open list is used  // push add node to the heap and return its reference
}
inline void SIPP::eraseNodeFromLists(SIPPNode* node)
{
    if (open_list.empty())
    { // we only have focal list
        focal_list.erase(node->focal_handle);
    }
    else if (focal_list.empty())
    {  // we only have open list
        open_list.erase(node->open_handle);
    }
    else
    { // we have both open and focal
        open_list.erase(node->open_handle);
        if (node->getFVal() <= w * min_f_val)
            focal_list.erase(node->focal_handle);
    }
}
void SIPP::releaseNodes()
{
    open_list.clear();
    focal_list.clear();
    for (auto & node_list : allNodes_table)
        for (auto n : node_list.second)
            delete n;
    allNodes_table.clear();
    for (auto n : useless_nodes)
        delete n;
    useless_nodes.clear();
}

// return true iff we the new node is not dominated by any old node-- algorithm 3
bool SIPP::dominanceCheck(SIPPNode* new_node)
{
    auto ptr = allNodes_table.find(new_node);  // hash table, one hash value can have multiple value
    if (ptr == allNodes_table.end())
        return true;
    for (auto & old_node : ptr->second)  // ptr->second: hash value
    {
        if (old_node->timestep <= new_node->timestep and
            old_node->num_of_conflicts <= new_node->num_of_conflicts)
        { // the new node is dominated by the old node, no need to generate new node
            return false;
        }
        else if (old_node->timestep >= new_node->timestep and
                old_node->num_of_conflicts >= new_node->num_of_conflicts) // the old node is dominated by the new node
        { // delete the old node
            if (old_node->in_openlist) // the old node has not been expanded yet
                eraseNodeFromLists(old_node); // delete it from open and/or focal lists
            else // the old node has been expanded already
                num_reopened++; //re-expand it
            useless_nodes.push_back(old_node);
            ptr->second.remove(old_node);
            num_generated--; // this is because we later will increase num_generated when we insert the new node into lists.
            return true;
        }
        else if(old_node->timestep < new_node->high_expansion and new_node->timestep < old_node->high_expansion)
        { // intervals overlap --> we need to split the node to make them disjoint
            if (old_node->timestep <= new_node->timestep)
            {
                assert(old_node->num_of_conflicts > new_node->num_of_conflicts);
                old_node->high_expansion = new_node->timestep;
            }
            else // i.e., old_node->timestep > new_node->timestep
            {
                assert(old_node->num_of_conflicts <= new_node->num_of_conflicts);
                new_node->high_expansion = old_node->timestep;
            }
        }
    }
    return true;
}

void SIPP::rosVisualization(int iteration, const Path& path)
{
    if (!myros || !marker_pub) return;

    visualization_msgs::MarkerArray marker_array;
    int width = instance.num_of_cols;
    int height = instance.num_of_rows;
    int unique_id = 100000;

    // ✅ 1. START & GOAL markers
    marker_array.markers.push_back(makeMarker(start_location % width, start_location / width, 1.0, 0.0, 0.0, "start", 0.5, 1.2));
    marker_array.markers.push_back(makeMarker(goal_location % width, goal_location / width, 0.6, 0.0, 1.0, "goal", 0.5, 1.2));

    // ✅ 2. Node EXPANSIONS (open/closed)
    for (const auto& pair : allNodes_table) {
        for (const auto& node : pair.second) {
            int x = node->location % width;
            int y = node->location / width;
            marker_array.markers.push_back(makeMarker(
                x, y,
                node->in_openlist ? 0.0f : 1.0f,  // R
                node->in_openlist ? 1.0f : 1.0f,  // G
                node->in_openlist ? 0.0f : 0.0f,  // B
                node->in_openlist ? "open" : "closed"
            ));
        }
    }

    // ✅ 3. Final path
    if (!path.empty()) {
        visualization_msgs::Marker line;
        line.header.frame_id = "map";
        line.header.stamp = ros::Time::now();
        line.ns = "final_path";
        line.id = unique_id++;
        line.type = visualization_msgs::Marker::LINE_STRIP;
        line.action = visualization_msgs::Marker::ADD;
        line.scale.x = 0.2;
        line.color.r = 1.0;
        line.color.g = 0.4;
        line.color.b = 0.0;
        line.color.a = 1.0;
        line.lifetime = ros::Duration(0.5);

        for (const auto& step : path) {
            geometry_msgs::Point p;
            p.x = (step.location % width) + 0.5;
            p.y = (step.location / width) + 0.5;
            p.z = 0.2;
            line.points.push_back(p);
        }

        marker_array.markers.push_back(line);
    }

    // 📤 Publish lightweight update
    marker_pub.publish(marker_array);
    ros::spinOnce();
    ros::Duration(0.05).sleep();
}