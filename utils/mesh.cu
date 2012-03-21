// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cutl/blob/master/LICENSE

#include "utils/mesh.h"

TriangleArray load_from_ply(const std::string & filename) {
    std::ifstream file;
    file.open(filename.c_str());
    std::string line;
    bool data = false;

    int n_vertices = 0;
    int n_faces = 0;

    std::vector<float3> vertices, normals;
    std::vector<std::vector<int> > faces;
    std::string temp;
    while(file.good()) {
        getline(file, line);
        std::istringstream sstr(line);
        if(!data) {
            if(line.size() > 13 && line.compare(0,14,"element vertex") == 0) {
                sstr >> temp >> temp >> n_vertices;
            }
            if(line.size() > 11 && line.compare(0,12,"element face") == 0) {
                sstr >> temp >> temp >> n_faces;
            }
            if(line.size() > 9 && line.compare(0,10,"end_header") == 0) {
                data = true;
            }
        } else { // parse data
            if(n_vertices > 0) {
                float3 vertex, normal;
                sstr >> vertex.x >> vertex.y >> vertex.z
                     >> normal.x >> normal.y >> normal.z;
                vertices.push_back(vertex);
                normals.push_back(normal);
                n_vertices--;
            } else {
                int temp;
                int face[3];
                sstr >> temp >> face[0] >> face[1] >> face[2];
                std::vector<int> ff;
                for(int i = 0; i < 3; ++i)
                    ff.push_back(face[i]);
                faces.push_back(ff);

            }
        }
    }

    std::vector<Triangle> tris;
    for(int f = 0; f < faces.size(); ++f) {
        Triangle tri;
        for(int i = 0; i < 3; ++i) {
            tri.v[i].f3.vec = vertices[faces[f][i]];
            tri.n[i].f3.vec = normals[faces[f][i]];
        }
        tris.push_back(tri);
    }

    return TriangleArray(tris);
}
