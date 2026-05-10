// RUN: spmcc --scr-chunk-parallel %s | FileCheck %s

// ============================================================================
// SCR-PARALLELIZABLE CASES - pop-only loops with independent computation
// ============================================================================

// Test 1: Pop with disjoint store - should tile + parallelize inner loop
// CHECK-LABEL: @pop_disjoint_store
// CHECK: affine.for
// CHECK: affine.parallel
// CHECK: spmc.pop_front
// CHECK: affine.store
func.func @pop_disjoint_store(%q: !spmc.queue<i32, 16>,
                               %out: memref<100xi32>) {
    affine.for %i = 0 to 100 {
        %rc, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)
        affine.store %v, %out[%i] : memref<100xi32>
    }
    return
}

// Test 2: Pop with pure arithmetic and disjoint store
// CHECK-LABEL: @pop_with_compute
// CHECK: affine.for
// CHECK: affine.parallel
// CHECK: spmc.pop_front
func.func @pop_with_compute(%q: !spmc.queue<i32, 16>,
                             %out: memref<100xi32>) {
    affine.for %i = 0 to 100 {
        %rc, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)
        %c2 = arith.constant 2 : i32
        %r = arith.muli %v, %c2 : i32
        affine.store %r, %out[%i] : memref<100xi32>
    }
    return
}

// ============================================================================
// NON-PARALLELIZABLE CASES - should remain affine.for
// ============================================================================

// Test 3: Push loop - single producer, not safe for parallelization
// CHECK-LABEL: @push_not_parallelized
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @push_not_parallelized(%q: !spmc.queue<i32, 16>) {
    affine.for %i = 0 to 100 {
        %v = arith.index_cast %i : index to i32
        %rc = "spmc.push_back"(%q, %v) : (!spmc.queue<i32, 16>, i32) -> i1
    }
    return
}

// Test 4: Both push and pop - push blocks parallelization
// CHECK-LABEL: @pop_and_push_not_parallelized
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @pop_and_push_not_parallelized(%qin: !spmc.queue<i32, 16>,
                                          %qout: !spmc.queue<i32, 16>) {
    affine.for %i = 0 to 100 {
        %rc, %v = "spmc.pop_front"(%qin) : (!spmc.queue<i32, 16>) -> (i1, i32)
        %rc2 = "spmc.push_back"(%qout, %v) : (!spmc.queue<i32, 16>, i32) -> i1
    }
    return
}

// Test 5: Pop with iter_args - loop-carried dependency blocks parallelization
// CHECK-LABEL: @pop_with_iter_args
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @pop_with_iter_args(%q: !spmc.queue<i32, 16>) -> i32 {
    %c0 = arith.constant 0 : i32
    %sum = affine.for %i = 0 to 100 iter_args(%acc = %c0) -> i32 {
        %rc, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)
        %new_acc = arith.addi %acc, %v : i32
        affine.yield %new_acc : i32
    }
    return %sum : i32
}

// Test 6: Pop with conflicting memory writes (all write to same index)
// CHECK-LABEL: @pop_with_memory_conflict
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @pop_with_memory_conflict(%q: !spmc.queue<i32, 16>,
                                     %out: memref<1xi32>) {
    affine.for %i = 0 to 100 {
        %rc, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)
        affine.store %v, %out[0] : memref<1xi32>
    }
    return
}
