#include <ostream>
#include <vector_types.h>
#include <cutil_math.h>
#include "kdtree.h"
#include "utils/primitives.h"
#include "utils/mesh.h"

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cerr << "usage: export_nodes small_cell_size" << std::endl;
        exit(1);
    }

    int small_size = atoi(argv[1]);
    cukd::TriangleArray triarr = cukd::load_from_ply("test/dragon/dragon_vrip_res2.ply");
    triarr.compute_aabbs();

    std::cout << "Creating k-d tree for " << triarr.size() << " triangles" 
              << std::endl;
    std::cout << "Keeping small nodes of maximal size " << small_size
              << std::endl;

    cukd::AABB root_aabb;
    root_aabb.minimum = make_float4(-3,-3,-3,0);
    root_aabb.maximum = make_float4(3,3,3,0);

    cutl::Timer full("creation time: ");

    cukd::KDTree kdtree(root_aabb, triarr, small_size);

    full.start();
    kdtree.create();
    full.stop();
    full.print();

//    cukd::KDTreeHost kdh = kdtree.to_host();
//    std::cout << "generated splits: " << kdh.split_position.size() << std::endl;
//    std::cout << "small nodes:      " << kdh.small_node_aabbs.size() << std::endl;
}
