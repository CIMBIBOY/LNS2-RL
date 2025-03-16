// mylns2_tester.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <string>
#include <sstream>
#include <filesystem>  // Requires C++17
#include <opencv2/opencv.hpp>

#include "./inc/mylns2.h"
#include "./inc/SingleAgentSolver.h"

namespace fs = std::filesystem;

// Utility: create a 64x64 map with random obstacles.
std::vector<std::vector<int>> createMap64x64(double obstacleProb = 0.1) {
    const int SIZE = 64;
    std::vector<std::vector<int>> map(SIZE, std::vector<int>(SIZE, 0));
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            double p = static_cast<double>(std::rand()) / RAND_MAX;
            if (p < obstacleProb) {
                map[r][c] = 1; // 1 represents an obstacle.
            }
        }
    }
    return map;
}

// Utility: Create 5 fixed start/goal pairs.
void createAgentPositions(std::vector<std::pair<int,int>> &starts,
                          std::vector<std::pair<int,int>> &goals) {
    starts.clear();
    goals.clear();
    // These positions must be within [0, 63]
    starts.push_back({2, 2});    goals.push_back({10, 10});
    starts.push_back({3, 50});   goals.push_back({10, 40});
    starts.push_back({50, 10});  goals.push_back({55, 55});
    starts.push_back({30, 30});  goals.push_back({2, 60});
    starts.push_back({60, 5});   goals.push_back({5, 60});
}

// Global visualization function for final (global) solution.
void dumpGlobalVisualization(const std::vector<std::vector<int>> &map,
                       int gridSize,
                       const std::vector<std::pair<int,int>> &agentStarts,
                       const std::vector<std::pair<int,int>> &agentGoals,
                       const std::vector<std::vector<std::pair<int,int>>> &paths,
                       int iteration)
{
    int scale = 10;
    int height = gridSize, width = gridSize;
    cv::Mat vis(height * scale, width * scale, CV_8UC3, cv::Scalar(255,255,255));

    // Assert that the image dimensions are as expected.
    assert(vis.rows == height * scale && vis.cols == width * scale);

    // Draw obstacles.
    for (int r = 0; r < height; r++){
        for (int c = 0; c < width; c++){
            if (map[r][c] == 1) {
                cv::rectangle(vis,
                              cv::Point(c * scale, r * scale),
                              cv::Point((c+1) * scale - 1, (r+1) * scale - 1),
                              cv::Scalar(0,0,0), cv::FILLED);
            }
        }
    }
    
    // Draw goals as blue circles.
    for (const auto &g : agentGoals) {
        int r = g.first, c = g.second;
        cv::circle(vis, cv::Point(c*scale + scale/2, r*scale + scale/2),
                   scale/3, cv::Scalar(255,0,0), cv::FILLED); // Blue (BGR)
    }
    
    // Draw agents as red circles.
    for (const auto &a : agentStarts) {
        int r = a.first, c = a.second;
        cv::circle(vis, cv::Point(c*scale + scale/2, r*scale + scale/2),
                   scale/3, cv::Scalar(0,0,255), cv::FILLED); // Red (BGR)
    }
    
    // Draw each agent’s computed path as a yellow line.
    for (const auto &path : paths) {
        if (path.size() < 2) continue;
        for (size_t i = 1; i < path.size(); i++) {
            int r0 = path[i-1].first, c0 = path[i-1].second;
            int r1 = path[i].first, c1 = path[i].second;
            cv::line(vis,
                     cv::Point(c0*scale + scale/2, r0*scale + scale/2),
                     cv::Point(c1*scale + scale/2, r1*scale + scale/2),
                     cv::Scalar(0,255,255), 1, cv::LINE_AA);
            cv::circle(vis,
                       cv::Point(c0*scale + scale/2, r0*scale + scale/2),
                       scale/6, cv::Scalar(0,255,255), cv::FILLED);
        }
        int r_last = path.back().first, c_last = path.back().second;
        cv::circle(vis,
                   cv::Point(c_last*scale + scale/2, r_last*scale + scale/2),
                   scale/6, cv::Scalar(0,255,255), cv::FILLED);
    }
    
    // Create output folder if it doesn't exist.
    std::string folder = "./global_visualizations/";
    if (!fs::exists(folder)) {
        fs::create_directory(folder);
    }
    iteration += 2;
    std::stringstream ss;
    ss << folder << "iter_" << iteration << ".png";
    std::string filename = ss.str();
    bool saved = cv::imwrite(filename, vis);
    assert(saved && "Failed to save global visualization image");
    std::cout << "Global visualization saved to " << filename << "\n";
}

///////////////////////////
// Main tester function.
int main() {
    std::srand(420); // For reproducibility

    // 1. Create the 64x64 map.
    auto map_data = createMap64x64(0.3); // 10% obstacle probability.
    int gridSize = 64;
    
    // 2. Create 5 agent start/goal pairs.
    std::vector<std::pair<int,int>> starts, goals;
    createAgentPositions(starts, goals);
    int num_agents = 5;
    
    // 3. Create an Instance.
    Instance instance(map_data, starts, goals, num_agents, gridSize);
    assert(instance.map_size == gridSize * gridSize && "Instance map size should equal gridSize^2");
    
    // 4. Instantiate LNS2.
    MyLns2 lns2_model(123, map_data, starts, goals, num_agents, gridSize);

    // 5. Run initial planning.
    lns2_model.init_pp();
    // Assert that at least one agent’s path is non-empty.
    assert(!lns2_model.vector_path.empty() && "No paths computed in initial planning!");
    
    // 6. Print the initial solution.
    std::cout << "=== Initial LNS2+SIPP solution ===\n";
    for (size_t ag = 0; ag < lns2_model.vector_path.size(); ag++) {
        std::cout << "Agent " << ag << ": ";
        for (auto &pt : lns2_model.vector_path[ag]) {
            std::cout << "(" << pt.first << "," << pt.second << ") -> ";
        }
        std::cout << "[GOAL]\n";
    }
    
    // 7. Dump a global visualization of the current solution.
    dumpGlobalVisualization(map_data, gridSize, starts, goals, lns2_model.vector_path, 0);
    
    // 8. Simulate a partial replan for agents 1 and 3.
    std::vector<int> subset = {1, 3};
    int collisions = lns2_model.calculate_sipps(subset);
    std::cout << "\nAfter partial replan for agents {1,3}, collisions found = " << collisions << "\n";
    
    for (size_t i = 0; i < subset.size(); i++) {
        int ag_id = subset[i];
        std::cout << "Agent " << ag_id << " new path:\n   ";
        if (i < lns2_model.sipps_path.size()) {
            for (auto &pt : lns2_model.sipps_path[i]) {
                std::cout << "(" << pt.first << "," << pt.second << ") -> ";
            }
            std::cout << "[GOAL]\n";
        } else {
            std::cout << "No new path computed.\n";
        }
    }
    
    // 9. Overwrite corresponding global paths.
    for (size_t i = 0; i < subset.size(); i++) {
        int ag_id = subset[i];
        if (i < lns2_model.sipps_path.size()) {
            lns2_model.vector_path[ag_id] = lns2_model.sipps_path[i];
        }
    }
    
    std::cout << "\n=== Final solution after partial replan ===\n";
    for (size_t ag = 0; ag < lns2_model.vector_path.size(); ag++) {
        std::cout << "Agent " << ag << ": ";
        for (auto &pt : lns2_model.vector_path[ag]) {
            std::cout << "(" << pt.first << "," << pt.second << ") -> ";
        }
        std::cout << "[GOAL]\n";
    }
    
    // 10. Dump a final global visualization.
    dumpGlobalVisualization(map_data, gridSize, starts, goals, lns2_model.vector_path, 1);
    
    std::cout << "Test complete.\n";
    return 0;
}