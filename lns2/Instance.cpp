#include<boost/tokenizer.hpp>
#include <algorithm>    // std::shuffle
#include"Instance.h"


Instance::Instance( const vector<vector<int>>& obs_map, const vector<pair<int,int>>& start_poss, const vector<pair<int,int>>& goal_poss,int num_of_agents,int num_of_rows):
        num_of_agents(num_of_agents),num_of_rows(num_of_rows),num_of_cols(num_of_rows) 
{
    /* const vector<vector<vector<int>>>& grid_sequence - ,dynamic_map_seq(grid_sequence)
    std::cout << "[INSTANCE] Initializing with " << num_of_agents << " agents\n";
    std::cout << "[INSTANCE] Grid sequence size: " << grid_sequence.size()
              << ", frame shape: (" << grid_sequence[0].size() << ", " << grid_sequence[0][0].size() << ")\n";
    */

    loadMap(obs_map); // grid_sequence[0])
    loadAgents(start_poss,goal_poss);

}

bool Instance::loadMap(const vector<vector<int>>& obs_map)
{
    map_size = num_of_cols * num_of_rows;
    my_map.resize(map_size, false);

    std::cout << "[LOADMAP] Loading static obstacle map...\n";
    int obs_count = 0;
    for (int i = 0; i < num_of_rows; i++) {
        for (int j = 0; j < num_of_cols; j++) {
            bool is_obs = (obs_map[i][j] != 0);
            my_map[linearizeCoordinate(i, j)] = is_obs;
            if (is_obs) obs_count++;
        }
    }
    std::cout << "[LOADMAP] Obstacle count: " << obs_count << "\n";
    return true;
}

bool Instance::loadAgents(const vector<pair<int,int>>& start_poss,
    const vector<pair<int,int>>& goal_poss)
{
    start_locations.resize(num_of_agents);
    goal_locations.resize(num_of_agents);

    std::cout << "[LOADAGENTS] Loading agent start and goal positions...\n";
    for (int i = 0; i < num_of_agents; i++) {
        int sx = start_poss[i].first;
        int sy = start_poss[i].second;
        int gx = goal_poss[i].first;
        int gy = goal_poss[i].second;
        start_locations[i] = linearizeCoordinate(sx, sy);
        goal_locations[i] = linearizeCoordinate(gx, gy);

        /*
        std::cout << "  → Agent " << i
        << " start=(" << sx << "," << sy << ") val=" << dynamic_map_seq[0][sx][sy]
        << " | goal=(" << gx << "," << gy << ") val=" << dynamic_map_seq[0][gx][gy]
        << std::endl;
        */
    }
    return true;
}


list<int> Instance::getNeighbors(int curr) const  // get truely moveable agent
{
	list<int> neighbors;
	int candidates[4] = {curr + 1, curr - 1, curr + num_of_cols, curr - num_of_cols};  // right, left, up,down
	for (int next : candidates)  // for next in candidates
	{
		if (validMove(curr, next))
			neighbors.emplace_back(next);
	}
	return neighbors;
}

bool Instance::isRadarZone(int timestep, int row, int col) const {
    if (timestep >= dynamic_map_seq.size()) return false;
    return dynamic_map_seq[timestep][row][col] == 2;
}
