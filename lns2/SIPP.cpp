#include "SIPP.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
namespace fs = std::filesystem;

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
    std::cout << "SIPP::findPath - Starting" << std::endl;
    std::cout << "  start_location: " << start_location << std::endl;
    std::cout << "  goal_location: " << goal_location << std::endl;
    
    reset();
    std::cout << "  After reset()" << std::endl;
    
    ReservationTable reservation_table(constraint_table, goal_location);
    std::cout << "  ReservationTable created" << std::endl;
    
    Path path;
    std::cout << "  Path initialized" << std::endl;
    
    std::cout << "  Getting first safe interval for location: " << start_location << std::endl;
    Interval interval = reservation_table.get_first_safe_interval(start_location);
    std::cout << "  First safe interval: [" << get<0>(interval) << ", " << get<1>(interval) << "]" << std::endl;
    
    if (get<0>(interval) > 0) {
        std::cout << "  Cannot hold start position at beginning. Returning empty path." << std::endl;
        return path;
    }
    
    std::cout << "  Getting holding time" << std::endl;
    auto holding_time = constraint_table.getHoldingTime(constraint_table.length_min);
    std::cout << "  holding_time: " << holding_time << std::endl;
    
    std::cout << "  Getting last collision timestep for goal: " << goal_location << std::endl;
    auto last_target_collision_time = constraint_table.getLastCollisionTimestep(goal_location);
    std::cout << "  last_target_collision_time: " << last_target_collision_time << std::endl;
    
    std::cout << "  Calculating heuristic for start location: " << start_location << std::endl;
    std::cout << "  my_heuristic size: " << my_heuristic.size() << std::endl;
    if (start_location >= 0 && start_location < my_heuristic.size()) {
        std::cout << "  my_heuristic[start_location]: " << my_heuristic[start_location] << std::endl;
    } else {
        std::cout << "  ERROR: start_location out of bounds for my_heuristic!" << std::endl;
        return path; // Avoid the crash by returning early
    }
    
    auto h = max(max(my_heuristic[start_location], holding_time), last_target_collision_time + 1);
    std::cout << "  h value: " << h << std::endl;
    
    std::cout << "  Creating start node" << std::endl;
    std::cout << "  interval tuple values: [" << get<0>(interval) << ", " 
              << get<1>(interval) << ", " << get<2>(interval) << "]" << std::endl;
    
    auto start = new SIPPNode(start_location, 0, h, nullptr, 0, get<1>(interval), get<1>(interval),
                              get<2>(interval), get<2>(interval));
    std::cout << "  Start node created" << std::endl;
    
    std::cout << "  Pushing start node to focal" << std::endl;
    pushNodeToFocal(start);
    std::cout << "  Start node pushed to focal" << std::endl;

    int iteration = 0; // iteration counter for visualization
    const int MAX_ITERATIONS = 4000;
    while (!focal_list.empty())  // algorithm 1
    {   
        // Check if we've exceeded max iterations
        if (iteration >= MAX_ITERATIONS) {
            std::cout << "SIPP::findPath - FAILED: No path found from " 
                    << start_location << " to " << goal_location 
                    << " after " << MAX_ITERATIONS << " iterations" << std::endl;
            break; // Exit the search loop
        }

        // Dump visualization every 50 iterations.
        if (iteration % 50 == 0) {
            dumpVisualization(iteration);
        }
        iteration++;

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
    }  // end while loop

    // At the end of the method, before returning:
    if (path.empty()) {
        std::cout << "SIPP::findPath - FAILED: No path found from " << start_location 
                  << " to " << goal_location << std::endl;
    } else {
        std::cout << "SIPP::findPath - SUCCESS: Path found with " 
                  << path.size() << " steps" << std::endl;
        std::cout << "  Final location: " << path.back().location << std::endl;
    }

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

void SIPP::dumpVisualization(int iteration)
{
    // Create visualization canvas
    int scale = 10;
    int width = 64;  // assuming 64x64 grid
    int height = 64;
    cv::Mat vis(height * scale, width * scale, CV_8UC3, cv::Scalar(255,255,255));
    
    // Convert locations to x,y coordinates
    int start_x = start_location % width;
    int start_y = start_location / width;
    int goal_x = goal_location % width;
    int goal_y = goal_location / width;
    
    // Draw grid lines for better visibility
    for (int i = 0; i <= height; i++) {
        cv::line(vis, cv::Point(0, i * scale), cv::Point(width * scale, i * scale), 
                 cv::Scalar(220, 220, 220), 1);
    }
    for (int j = 0; j <= width; j++) {
        cv::line(vis, cv::Point(j * scale, 0), cv::Point(j * scale, height * scale), 
                 cv::Scalar(220, 220, 220), 1);
    }
    
    // Draw explored nodes (closed nodes) with color indicating expansion order
    std::vector<std::pair<int, int>> closed_nodes; // location, timestep
    for (const auto& nodes_pair : allNodes_table) {
        for (const auto& node : nodes_pair.second) {
            if (!node->in_openlist && node->timestep > 0) {
                closed_nodes.push_back({node->location, node->timestep});
            }
        }
    }
    
    // Sort nodes by timestep to visualize expansion order
    std::sort(closed_nodes.begin(), closed_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Color nodes by their relative expansion order
    for (size_t i = 0; i < closed_nodes.size(); i++) {
        int location = closed_nodes[i].first;
        int x = location % width;
        int y = location / width;
        
        // Calculate color based on position in the sequence (yellow to red gradient)
        double ratio = static_cast<double>(i) / closed_nodes.size();
        cv::Scalar color(0, 255 * (1.0 - ratio), 255 * (1.0 - ratio * 0.5));
        
        cv::circle(vis, cv::Point(x*scale + scale/2, y*scale + scale/2),
                   scale/3, color, cv::FILLED);
    }
    
    // Draw the frontier (open nodes)
    std::vector<const SIPPNode*> open_nodes;
    for (const auto& nodes_pair : allNodes_table) {
        for (const auto& node : nodes_pair.second) {
            if (node->in_openlist) {
                open_nodes.push_back(node);
            }
        }
    }
    
    // Sort open nodes by f-value
    std::sort(open_nodes.begin(), open_nodes.end(), 
              [](const auto* a, const auto* b) { return a->getFVal() < b->getFVal(); });
    
    // Draw open nodes with color indicating f-value (best to worst: bright green to dark green)
    for (size_t i = 0; i < open_nodes.size(); i++) {
        int location = open_nodes[i]->location;
        int x = location % width;
        int y = location / width;
        
        double ratio = static_cast<double>(i) / open_nodes.size();
        int green = 255 - static_cast<int>(150 * ratio);
        cv::Scalar color(0, green, 0);
        
        cv::circle(vis, cv::Point(x*scale + scale/2, y*scale + scale/2),
                   scale/3, color, cv::FILLED);
    }
    
    // Draw path from start to current best node (if available)
    if (!focal_list.empty()) {
        const LLNode* curr = focal_list.top();
        std::vector<int> path_locations;
        
        // Trace back the path
        while (curr != nullptr) {
            path_locations.push_back(curr->location);
            curr = curr->parent;
        }
        
        // Draw the path
        for (size_t i = 1; i < path_locations.size(); i++) {
            int from_loc = path_locations[i];
            int to_loc = path_locations[i-1];
            
            int from_x = from_loc % width;
            int from_y = from_loc / width;
            int to_x = to_loc % width;
            int to_y = to_loc / width;
            
            cv::line(vis, 
                     cv::Point(from_x*scale + scale/2, from_y*scale + scale/2),
                     cv::Point(to_x*scale + scale/2, to_y*scale + scale/2),
                     cv::Scalar(0, 165, 255), 2);  // Orange line
        }
        
        // Draw the current best node in the focal list
        int best_loc = focal_list.top()->location;
        int best_x = best_loc % width;
        int best_y = best_loc / width;
        cv::circle(vis, cv::Point(best_x*scale + scale/2, best_y*scale + scale/2),
                  scale/2, cv::Scalar(0, 255, 255), cv::FILLED);  // Yellow for current best
    }
    
    // Draw start and goal (drawn last to be on top)
    cv::circle(vis, cv::Point(start_x*scale + scale/2, start_y*scale + scale/2),
               scale/2, cv::Scalar(0, 0, 255), cv::FILLED);  // Start: red
    cv::putText(vis, "S", cv::Point(start_x*scale + scale/4, start_y*scale + 2*scale/3),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
                
    cv::circle(vis, cv::Point(goal_x*scale + scale/2, goal_y*scale + scale/2),
               scale/2, cv::Scalar(255, 0, 0), cv::FILLED);  // Goal: blue
    cv::putText(vis, "G", cv::Point(goal_x*scale + scale/4, goal_y*scale + 2*scale/3),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    
    // Add information text
    std::string iter_text = "Iteration: " + std::to_string(iteration);
    cv::putText(vis, iter_text, cv::Point(10, height*scale - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    std::string open_text = "Open: " + std::to_string(open_nodes.size());
    cv::putText(vis, open_text, cv::Point(10, height*scale - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    std::string closed_text = "Closed: " + std::to_string(closed_nodes.size());
    cv::putText(vis, closed_text, cv::Point(10, height*scale - 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    if (!focal_list.empty()) {
        std::string f_val_text = "Best f-value: " + std::to_string(focal_list.top()->getFVal());
        cv::putText(vis, f_val_text, cv::Point(10, height*scale - 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Create unique folder structure for visualizations
    std::string folder = "./sipps_visualizations/";
    if (!fs::exists(folder)) {
        fs::create_directory(folder);
    }
    
    std::string agent_folder = folder + "from" + std::to_string(start_location) + 
                               "to" + std::to_string(goal_location) + "/";
    if (!fs::exists(agent_folder)) {
        fs::create_directory(agent_folder);
    }
    
    // Save visualization with unique filename including iteration number
    std::string filename = agent_folder + "iter_" + std::to_string(iteration) + ".png";
    cv::imwrite(filename, vis);
    std::cout << "SIPP visualization saved to " << filename << std::endl;
    
    // Add a legend image for the first visualization
    if (iteration == 0) {
        cv::Mat legend(180, 240, CV_8UC3, cv::Scalar(255, 255, 255));
        int y_pos = 20;
        int step = 20;
        
        // Draw legend items
        cv::circle(legend, cv::Point(20, y_pos), 5, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::putText(legend, "Start", cv::Point(40, y_pos+5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        y_pos += step;
        
        cv::circle(legend, cv::Point(20, y_pos), 5, cv::Scalar(255, 0, 0), cv::FILLED);
        cv::putText(legend, "Goal", cv::Point(40, y_pos+5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        y_pos += step;
        
        cv::circle(legend, cv::Point(20, y_pos), 5, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(legend, "Open Nodes (by f-value)", cv::Point(40, y_pos+5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        y_pos += step;
        
        cv::circle(legend, cv::Point(20, y_pos), 5, cv::Scalar(255, 255, 0), cv::FILLED);
        cv::putText(legend, "Expanded Nodes (by time)", cv::Point(40, y_pos+5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        y_pos += step;
        
        cv::circle(legend, cv::Point(20, y_pos), 5, cv::Scalar(0, 255, 255), cv::FILLED);
        cv::putText(legend, "Current Best Node", cv::Point(40, y_pos+5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        y_pos += step;
        
        cv::line(legend, cv::Point(15, y_pos), cv::Point(25, y_pos), 
                 cv::Scalar(0, 165, 255), 2);
        cv::putText(legend, "Current Path", cv::Point(40, y_pos+5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        std::string legend_filename = agent_folder + "legend.png";
        cv::imwrite(legend_filename, legend);
    }
}