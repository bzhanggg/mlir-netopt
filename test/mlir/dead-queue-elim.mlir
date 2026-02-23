// TODO: use the actual FileCheck test framework

// 1. simple dead queue case
func.func @dead_queue_is_eliminated() -> () {
    %unused_queue = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    func.return
}

// 2. simple push queue case
func.func @pushed_queue_is_not_eliminated(%val: i32) -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    "spmc.push_back"(%q, %val) : (!spmc.queue<i32, 16>, i32) -> ()

    func.return
}

// 3. simple pop queue case
func.func @popped_queue_is_not_eliminated() -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> i32

    func.return
}

// 4. queue with multiple uses should not be eliminated
func.func @queue_with_multiple_use(%val: i32) -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    "spmc.push_back"(%q, %val) : (!spmc.queue<i32, 16>, i32) -> ()
    "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> i32
    func.return
}

// 5. queue only in result operand should not be eliminated
func.func @queue_only_in_result_operand(%val: i32) -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %result = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> i32
    func.return
}

// 3. queue should be eliminated across return flows if all unused
func.func @queue_in_control_flow(%cond: i1) -> () {
    cf.cond_br %cond, ^bb1, ^bb2
    ^bb1:
        %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
        func.return
    ^bb2:
        func.return
}

// 4. multiple independent dead queues, q1 and q2 should be eliminated
func.func @multiple_dead_queues(%val: i32) -> () {
    %q1 = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %q2 = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    %q3 = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    "spmc.push_back"(%q3, %val) : (!spmc.queue<i32, 16>, i32) -> ()
    func.return
}


// 5. queue passed as function arg should not be eliminated
func.func @queue_is_function_parameter(%q: !spmc.queue<i32, 16>) -> () {
    func.return
}

// 6. queue passed as function return should not be eliminated
func.func @queue_is_function_return() -> !spmc.queue<i32, 16> {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    func.return %q : !spmc.queue<i32, 16>
}
