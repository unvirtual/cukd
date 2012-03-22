#include <ostream>
#include <vector_types.h>
#include <cutil_math.h>
#include <set>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include "utils/primitives.h"
#include "kdtree.h"
#include "utils/mesh.h"

unsigned int pixel_from_color(const float3 & color) {
    return (unsigned int) (  ((int) round(color.x) << 16)
                           + ((int) round(color.y) << 8)
                           + round(color.z));
}

void put_pixel(const float3 & color, int x, int y, int width, unsigned int* buffer) {
    buffer[x + y*width] = pixel_from_color(color);
}

void write_to_bmp(int width, int height, std::vector<int> & hits, 
                  std::vector<int> cost) {
   std::string filename = "test.bmp";

   unsigned int* data_buffer = new unsigned int[width * height] ;
   for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; ++j) {
            int index = width*j + i;
            if(hits[width*j + i] != -1)
                put_pixel(make_float3(cost[index],0,1.f), i, j, width, data_buffer);
            else
                put_pixel(make_float3(0,cost[index],0), i, j, width, data_buffer);
        }
    }

    SDL_Surface * image = SDL_CreateRGBSurfaceFrom(data_buffer, width, height,
            32, width*4, 0x00ff0000, 0x0000ff00, 0x000000ff, 0);

   SDL_SaveBMP(image, filename.c_str());
   SDL_FreeSurface(image);
}


void get_rays(int width, int height, float xmin, float xmax,
              float ymin, float ymax, std::vector<Ray> & rays_h) {
    Ray ray;
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            float x =  (xmax - xmin)/((float) width)*j - xmax;
            float y = - (ymax - ymin)/((float) height)*i + ymax;
            ray.direction = finite_ufloat4(make_ufloat4(0.f,0.f,1,0));
            ray.origin = make_ufloat4(x,y,-5,0);
            rays_h.push_back(ray);
        }
    }
}

int main(int argc, char* argv[]) {
    using namespace cukd;

    // create the tree
    std::cout << "Loading geometry ..." << std::endl;
    TriangleArray triarr = load_from_ply("test/dragon/dragon_vrip_res2.ply");
    triarr.compute_aabbs();

    std::cout << "Creating k-d tree for " << triarr.size() << " triangles"
              << std::endl;

    UAABB root_aabb;
    root_aabb.minimum = make_ufloat4(-3,-3,-10,0);
    root_aabb.maximum = make_ufloat4(3,3,1,0);

    std::cout << "    initializing tree.." << std::endl;
    KDTree kdtree(root_aabb, triarr, 64);

    std::cout << "    creating tree..." << std::endl;
    Timer full("total tree creation time: ");

    kdtree.create();
    kdtree.preorder();
    
    for(int i = 0; i < 10; ++i) {
        full.start();
        kdtree.clear(triarr);
        kdtree.create();
        kdtree.preorder();
        full.stop();
    }
    full.print();

    std::cout << "tree creation time average: " << full.get_ms()/10 << " ms" << std::endl;

    // run raytraversal and measure the time
    const int repetitions = 100;
    const int width = 1024, height = 864;
    int n_rays = width * height;

    std::vector<Ray> rays_h;
    float xmin = -0.1, xmax = 0.2;
    float ymin = -0.1, ymax = 0.2;
    get_rays(width, height, xmin, xmax, ymin, ymax, rays_h);

    RayArray rays(rays_h);

    DevVector<int> hits;
    DevVector<int> costs;
    hits.resize(n_rays);
    costs.resize(n_rays);

    Timer traversal_time("total traversal time: ");
    std::cout << "traversing the tree " << repetitions << " times" << std::endl;
    traversal_time.start();
    for(int i = 0; i < repetitions; ++i)
        kdtree.ray_bunch_traverse(width, height, rays, hits, costs);
    traversal_time.stop();
    traversal_time.print();
    std::cout << "Total number of rays: " << n_rays << std::endl;
    std::cout << "Rays/sec:   " << (repetitions * 1000.f * n_rays/traversal_time.get_ms()) << std::endl;
    std::cout << "frames/sec: " << (repetitions * 1000.f / traversal_time.get_ms()) << std::endl;

    // write a bmp with the traversal cost
    std::vector<int> result, costh;
    thrust::copy(hits.begin(), hits.end(), std::back_inserter(result));
    thrust::copy(costs.begin(), costs.end(), std::back_inserter(costh));

    write_to_bmp(width, height, result, costh);
}
