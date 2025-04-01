#include "common.h"

std::ostream& operator<<(std::ostream& os, const Path& path)
{
	for (const auto& state : path)
	{
		os << state.location << "\t"; // << "(" << state.is_single() << "),";
	}
	return os;
}

visualization_msgs::Marker makeMarker(
    int x, int y,
    float r, float g, float b,
    const std::string& ns,
    float scale_xy,
    float scale_z,
    float z_offset,
    int id)
{
    static int next_id = 0;

    visualization_msgs::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.id = (id < 0) ? next_id++ : id;
    m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose.position.x = x + 0.5;
    m.pose.position.y = y + 0.5;
    m.pose.position.z = z_offset;
    m.scale.x = scale_xy;
    m.scale.y = scale_xy;
    m.scale.z = scale_z;
    m.color.r = r;
    m.color.g = g;
    m.color.b = b;
    m.color.a = 1.0;
    m.lifetime = ros::Duration(1.0);  // can be shorter if needed
    return m;
}