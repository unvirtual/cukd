// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cutl/blob/master/LICENSE

#ifndef CUTL_DEV_STRUCTS_H
#define CUTL_DEV_STRUCTS_H

struct DevAABBArray {
    int length;
    UFloat4* minima, *maxima;
};

struct DevTriangleArray {
    int length;
    UFloat4 *v[3];
    UFloat4 *n[3];
    DevAABBArray aabbs;
};

struct DevRayArray {
    int length;
    UFloat4 *origins;
    UFloat4 *directions;
};

#endif  // CUTL_DEV_STRUCTS_H
