module {
  func.func @fold_and_cse(%arg0: i32) -> i32 {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = arith.addi %c2_i32, %c3_i32 : i32
    %1 = arith.muli %arg0, %0 : i32
    %2 = arith.muli %arg0, %0 : i32
    %3 = arith.addi %1, %2 : i32
    return %3 : i32
  }
}

