func.func @dead_queue_is_eliminated() -> () {
    %unused_queue = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    func.return
}

func.func @pushed_queue_is_not_eliminated(%val: i32) -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    "spmc.push_back"(%q, %val) : (!spmc.queue<i32, 16>, i32) -> ()

    func.return
}

func.func @popped_queue_is_not_eliminated() -> () {
    %q = "spmc.create"() {element=i32, capacity=16 : ui32} : () -> !spmc.queue<i32, 16>
    "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> i32

    func.return
}

// more test cases:

// 1. queue with multiple uses
// %q = spmc.create()
// spmc.push_back(%q, %val)
// spmc.pop_front(%q)

// 2. queue only in result operand
// %result = spmc.pop_front(%q)  // %q used even though %result might be dead

// 3. nested operations, control flow
// cf.cond_br %cond, ^bb1, ^bb2
// ^bb1:
//   %q = spmc.create()
//   func.return
// ^bb2:
//   func.return

// 4. multiple independent dead queues
// %q1 = spmc.create()
// %q2 = spmc.create()
// %q3 = spmc.create()
// spmc.push_back(%q3, %val)

// 5. queue passed as function arg should not be eliminated
// func.func @process(%q: !spmc.queue) -> () {
//   // %q is a parameter, not created here — should not be eliminated
// }

// 6. queue returned from function escapes
// func.func @create_queue() -> !spmc.queue {
//   %q = spmc.create()
//   func.return %q
// }

// 7. dead result but queue used
// %unused = spmc.pop_front(%q)  // result unused but %q still counts as used