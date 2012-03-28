cukd
----

k-d tree for triangle clouds in CUDA, based on "Real Time KD-Tree
Construction on Graphics Hardware" by Zhou et al.
(check out [these articles](http://unvirtual.github.com/tag/cukd/) for more info)

![Dragon cost](http://unvirtual.github.com/img/dragon_ray_traversal.jpg)

This is a working preliminary version with acceptable performance,
generating the tree for 200k triangles takes ~250ms on an Nvidia GTX
760 and computes 40 million rays/sec (/test/ray_traversal.cu)


   - Creates a compact preorder sorted tree representation and provides
     parallel while-while ray traversal and ray-triangle intersections.

   - Generates a k-d tree in a two-stage process:

      - Empty space removal and median node splitting
      - Surface Area Heuristics

