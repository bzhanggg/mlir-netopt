// ./build/bin/spmcc --pass-pipeline="builtin.module(func.func(affine-loop-tile{tile-size=512}))" ./test/mlir/affine-tile.mlir

module {
    func.func @tile_push_back() -> () {
        %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
        affine.for %i = 0 to 4096 {
            %v = arith.index_cast %i : index to i32
            %rc = "spmc.push_back"(%q, %v) : (!spmc.queue<i32, 16>, i32) -> i1
        }
        func.return
    }

    // !queue = !spmc.queue<i32, 16>
    // #map = affine_map<(d0) -> (d0)>
    // #map1 = affine_map<(d0) -> (d0 + 512)>
    // module {
    //     func.func @tile_push_back() {
    //         %0 = "spmc.create"() <{capacity = 16 : ui32, element = i32}> : () -> !queue
    //         affine.for %arg0 = 0 to 4096 step 512 {
    //         affine.for %arg1 = #map(%arg0) to #map1(%arg0) {
    //             %1 = arith.index_cast %arg1 : index to i32
    //             spmc.push_back %0, %1 : (!queue, i32)
    //         }
    //         }
    //         return
    //     }
    // }
}