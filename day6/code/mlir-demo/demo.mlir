// Day6 step1: minimal MLIR demo.
// Dialects in use: builtin (module), func, arith.
module {
  func.func @fold_and_cse(%a: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %sum = arith.addi %c2, %c3 : i32
    %x = arith.muli %a, %sum : i32
    %y = arith.muli %a, %sum : i32
    %z = arith.addi %x, %y : i32
    return %z : i32
  }
}
