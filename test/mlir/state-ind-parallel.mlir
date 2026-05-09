// RUN: spmcc --state-ind-parallel %s | FileCheck %s

// ============================================================================
// PARALLELIZABLE CASES - should become affine.parallel
// ============================================================================

// Test 1: Disjoint affine load/store indexed by IV
// CHECK-LABEL: @disjoint_memref_access
// CHECK: affine.parallel
// CHECK-NOT: affine.for
func.func @disjoint_memref_access(%A: memref<1024xf32>, %B: memref<1024xf32>) {
    affine.for %i = 0 to 1024 {
        %v = affine.load %A[%i] : memref<1024xf32>
        %r = arith.mulf %v, %v : f32
        affine.store %r, %B[%i] : memref<1024xf32>
    }
    return
}

// Test 2: Pure arithmetic loop (no memory ops)
// CHECK-LABEL: @pure_arithmetic_loop
// CHECK: affine.parallel
// CHECK-NOT: affine.for
func.func @pure_arithmetic_loop(%out: memref<256xi32>) {
    affine.for %i = 0 to 256 {
        %idx = arith.index_cast %i : index to i32
        %c2 = arith.constant 2 : i32
        %v = arith.muli %idx, %c2 : i32
        affine.store %v, %out[%i] : memref<256xi32>
    }
    return
}

// Test 3: Read-only loop with disjoint writes
// CHECK-LABEL: @read_only_shared_write_disjoint
// CHECK: affine.parallel
// CHECK-NOT: affine.for
func.func @read_only_shared_write_disjoint(%src: memref<512xf32>,
                                            %dst: memref<512xf32>,
                                            %scale: f32) {
    affine.for %i = 0 to 512 {
        %v = affine.load %src[%i] : memref<512xf32>
        %r = arith.mulf %v, %scale : f32
        affine.store %r, %dst[%i] : memref<512xf32>
    }
    return
}

// ============================================================================
// NON-PARALLELIZABLE CASES - should remain affine.for
// ============================================================================

// Test 4: Loop with SPMC push operation
// CHECK-LABEL: @spmc_push_not_parallel
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @spmc_push_not_parallel(%q: !spmc.queue<i32, 16>) {
    affine.for %i = 0 to 100 {
        %v = arith.index_cast %i : index to i32
        %rc = "spmc.push_back"(%q, %v) : (!spmc.queue<i32, 16>, i32) -> i1
    }
    return
}

// Test 5: Loop with SPMC pop operation
// CHECK-LABEL: @spmc_pop_not_parallel
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @spmc_pop_not_parallel(%q: !spmc.queue<i32, 16>,
                                  %out: memref<100xi32>) {
    affine.for %i = 0 to 100 {
        %rc, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)
        affine.store %v, %out[%i] : memref<100xi32>
    }
    return
}

// Test 6: Loop with iter_args (loop-carried accumulation)
// CHECK-LABEL: @iter_args_not_parallel
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @iter_args_not_parallel(%A: memref<1024xf32>) -> f32 {
    %c0 = arith.constant 0.0 : f32
    %sum = affine.for %i = 0 to 1024 iter_args(%acc = %c0) -> f32 {
        %v = affine.load %A[%i] : memref<1024xf32>
        %new_acc = arith.addf %acc, %v : f32
        affine.yield %new_acc : f32
    }
    return %sum : f32
}

// Test 7: Loop writing to same memref element (all iterations write to index 0)
// CHECK-LABEL: @same_element_write_not_parallel
// CHECK: affine.for
// CHECK-NOT: affine.parallel
func.func @same_element_write_not_parallel(%A: memref<1024xf32>,
                                            %out: memref<1xf32>) {
    affine.for %i = 0 to 1024 {
        %v = affine.load %A[%i] : memref<1024xf32>
        affine.store %v, %out[0] : memref<1xf32>
    }
    return
}
