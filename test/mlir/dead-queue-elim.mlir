// RUN: /Volumes/workplace/mlir-netsec/build/bin/spmcc --dead-queue-elimination %s | FileCheck %s

// Test 1: Simple dead queue case - should be eliminated
// CHECK-LABEL: @dead_queue_is_eliminated
// CHECK-NOT: spmc.create
func.func @dead_queue_is_eliminated() -> () {
    %unused_queue = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    func.return
}

// Test 2: Queue with push - should NOT be eliminated
// CHECK-LABEL: @pushed_queue_is_not_eliminated
// CHECK: spmc.create
// CHECK: spmc.push_back
func.func @pushed_queue_is_not_eliminated(%val: i32) -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %rc = "spmc.push_back"(%q, %val) : (!spmc.queue<i32, 16>, i32) -> i1

    func.return
}

// Test 3: Queue with pop - should NOT be eliminated
// CHECK-LABEL: @popped_queue_is_not_eliminated
// CHECK: spmc.create
// CHECK: spmc.pop_front
func.func @popped_queue_is_not_eliminated() -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %rc, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)

    func.return
}

// Test 4: Queue with multiple uses - should NOT be eliminated
// CHECK-LABEL: @queue_with_multiple_use
// CHECK: spmc.create
// CHECK: spmc.push_back
// CHECK: spmc.pop_front
func.func @queue_with_multiple_use(%val: i32) -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %rc1 = "spmc.push_back"(%q, %val) : (!spmc.queue<i32, 16>, i32) -> i1
    %rc2, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)
    func.return
}

// Test 5: Queue used as result operand - should NOT be eliminated
// CHECK-LABEL: @queue_only_in_result_operand
// CHECK: spmc.create
// CHECK: spmc.pop_front
func.func @queue_only_in_result_operand(%val: i32) -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %rc, %v = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> (i1, i32)
    func.return
}

// Test 6: Queue in control flow - should be eliminated if unused
// CHECK-LABEL: @queue_in_control_flow
// CHECK-NOT: spmc.create
func.func @queue_in_control_flow(%cond: i1) -> () {
    cf.cond_br %cond, ^bb1, ^bb2
    ^bb1:
        %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
        func.return
    ^bb2:
        func.return
}

// Test 7: Multiple dead queues (q1, q2 dead; q3 live) - q1 and q2 should be eliminated
// CHECK-LABEL: @multiple_dead_queues
// CHECK-NOT: spmc.create
// CHECK-NOT: spmc.create
// CHECK: spmc.create
// CHECK: spmc.push_back
func.func @multiple_dead_queues(%val: i32) -> () {
    %q1 = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %q2 = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %q3 = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %rc = "spmc.push_back"(%q3, %val) : (!spmc.queue<i32, 16>, i32) -> i1
    func.return
}

// Test 8: Queue as function parameter - should NOT be eliminated
// CHECK-LABEL: @queue_is_function_parameter
func.func @queue_is_function_parameter(%q: !spmc.queue<i32, 16>) -> () {
    func.return
}

// Test 9: Queue as function return - should NOT be eliminated
// CHECK-LABEL: @queue_is_function_return
// CHECK: spmc.create
// CHECK: func.return
func.func @queue_is_function_return() -> !spmc.queue<i32, 16> {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    func.return %q : !spmc.queue<i32, 16>
}
