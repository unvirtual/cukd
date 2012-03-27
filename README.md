cukd
----

k-d tree for triangle clouds in CUDA, based on "Real Time KD-Tree
Construction on Graphics Hardware" by Zhou et al.

Generates a k-d tree in a two-stage process:
    * Empty space removal and median node splitting
    * Surface Area Heuristics

![Dragon cost](http://unvirtual.github.com/img/dragon_ray_traversal.jpg)

Creates a compact preorder sorted tree representation and provides
parallel while-while ray traversal and ray-triangle intersections.

This is a working preliminary version with acceptable performance,
generating the tree for 200k triangles takes ~250ms on an Nvidia GTX
760 and is capable of traversing 40 million rays/sec for the scene
(check out the small application in /test)
