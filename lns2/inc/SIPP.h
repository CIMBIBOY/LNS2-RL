﻿#pragma once
#include <boost/functional/hash.hpp>
// #include <opencv2/opencv.hpp>
#include "SingleAgentSolver.h"
#include "ReservationTable.h"
#include <cstdlib>
#include <unistd.h>

// 🔁 Replace ROS with Rerun
#include <rerun.hpp>

class SIPPNode: public LLNode
{
public:
	// define a typedefs for handles to the heaps (allow up to quickly update a node in the heap)
	typedef boost::heap::pairing_heap< SIPPNode*, compare<SIPPNode::compare_node> >::handle_type open_handle_t;
	typedef boost::heap::pairing_heap< SIPPNode*, compare<SIPPNode::secondary_compare_node> >::handle_type focal_handle_t;
	open_handle_t open_handle;
	focal_handle_t focal_handle;
	int high_generation; // the upper bound with respect to generation
    int high_expansion; // the upper bound with respect to expansion
	bool collision_v;  // if there is a collision
    SIPPNode() : LLNode() {}
	SIPPNode(int loc, int g_val, int h_val, SIPPNode* parent, int timestep, int high_generation, int high_expansion,
	        bool collision_v, int num_of_conflicts) :  // timestep=interval.low
            LLNode(loc, g_val, h_val, parent, timestep, num_of_conflicts), high_generation(high_generation),
            high_expansion(high_expansion), collision_v(collision_v) {}
	// SIPPNode(const SIPPNode& other): LLNode(other), high_generation(other.high_generation), high_expansion(other.high_expansion),
        //                              collision_v(other.collision_v) {}
	~SIPPNode() {}

	void copy(const SIPPNode& other) // copy everything except for handles
    {
	    LLNode::copy(other);
        high_generation = other.high_generation;
        high_expansion = other.high_expansion;
        collision_v = other.collision_v;
    }
	// The following is used by for generating the hash value of a nodes
	struct NodeHasher
	{
		std::size_t operator()(const SIPPNode* n) const //  generate hash value from thw two property
		{
            size_t seed = 0;
            boost::hash_combine(seed, n->location);
            boost::hash_combine(seed, n->high_generation);
            return seed;
		}
	};

	// The following is used for checking whether two nodes are equal
	// we say that two nodes, s1 and s2, are equal if
	// both are non-NULL and agree on the id and timestep
	struct eqnode
	{
		bool operator()(const SIPPNode* n1, const SIPPNode* n2) const
		{
			return (n1 == n2) ||
			            (n1 && n2 && n1->location == n2->location &&
				        n1->wait_at_goal == n2->wait_at_goal &&
				        n1->is_goal == n2->is_goal &&
                         n1->high_generation == n2->high_generation);
                        //max(n1->timestep, n2->timestep) <
                        //min(get<1>(n1->interval), get<1>(n2->interval))); //overlapping time intervals
		}
	};
};

class SIPP: public SingleAgentSolver
{
public:

    // find path by SIPP
	// Returns a shortest path that satisfies the constraints of the give node  while
	// minimizing the number of internal conflicts (that is conflicts with known_paths for other agents found so far).
	// lowerbound is an underestimation of the length of the path in order to speed up the search.
    //Path findOptimalPath(const PathTable& path_table) {return Path(); } // TODO: To implement
    //Path findOptimalPath(const ConstraintTable& constraint_table, const PathTableWC& path_table);
    Path findPath(const ConstraintTable& constraint_table); // return A path that minimizes collisions, breaking ties by cost
	// New visualization routine:
	// Add this to the SIPP class definition
	void rerunVisualization(int iteration, const Path& path);
	// Data members used in visualization (you may adjust these as needed):
    // "instance" is already available from SingleAgentSolver (if not, include it here)
    // Here we add a simple structure for world if not already defined.
    struct World {
        std::vector<int> agents_poss; // current positions (linearized indices)
        // Other members can be added as needed.
    } world;

    // The current best path found so far (as linearized indices).
    std::vector<int> current_best_path;
    // A container holding pointers to nodes currently in the focal list.
    std::list<LLNode*> focal_list_container;
    
    // Other visualization-related members...
    int num_expanded;  // for example, number of nodes expanded

	SIPP(const Instance& instance, int agent,const vector<int>& start_locations,const vector<int>& goal_locations):
		SingleAgentSolver(instance, agent,start_locations,goal_locations) {}

	// Visualization flags
    bool use_rerun = true;

private:
	// define typedefs and handles for heap
	typedef boost::heap::pairing_heap< SIPPNode*, boost::heap::compare<LLNode::compare_node> > heap_open_t;  // data structure
	typedef boost::heap::pairing_heap< SIPPNode*, boost::heap::compare<LLNode::secondary_compare_node> > heap_focal_t;
	heap_open_t open_list;
	heap_focal_t focal_list;

	// define typedef for hash_map
	typedef boost::unordered_map<SIPPNode*, list<SIPPNode*>, SIPPNode::NodeHasher, SIPPNode::eqnode> hashtable_t; // key,value, hash function, compare function
    hashtable_t allNodes_table;  // compare with map, hash map gerate hash value based on the input key, thus the finding speed is much faster
    list<SIPPNode*> useless_nodes;
    // Path findNoCollisionPath(const ConstraintTable& constraint_table);

    void updatePath(const LLNode* goal, std::vector<PathEntry> &path);

    inline void pushNodeToFocal(SIPPNode* node);
    inline void eraseNodeFromLists(SIPPNode* node);
	void releaseNodes();
    bool dominanceCheck(SIPPNode* new_node);
};

