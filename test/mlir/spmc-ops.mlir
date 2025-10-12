%n = arith.constant 8 : i32
%val = arith.constant 42 : i32

%q = "spmc.create"() {element=i32, capacity=16} : () -> !spmc.queue<i32, 16>

"spmc.push_back"(%q, %val) : (!spmc.queue<i32, 16>, i32) -> ()

%elt = "spmc.pop_front"(%q) : (!spmc.queue<i32, 16>) -> i32
