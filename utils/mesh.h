// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cutl/blob/master/LICENSE

#ifndef CUTL_MESH_H
#define CUTL_MESH_H

#include "utils/primitives.h"

// Simple ply mesh loader. Doesn't support quads and requires vertex
// and normal data only.
TriangleArray load_from_ply(const std::string & filename);

#endif  // CUTL_MESH_H
