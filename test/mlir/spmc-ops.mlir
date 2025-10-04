%n = arith.constant 8 : i32
%val = arith.constant 42 : i32

%q = "spmc.create"(%n) : (i32) -> !spmc.queue<i32>

"spmc.push"(%q, %val) : (!spmc.queue<i32>, i32) -> ()

%elt = "spmc.pop"(%q) : (!spmc.queue<i32>) -> (i32)