module {
  func.func @fold_and_cse(%arg0: i32) -> i32 {
    %c5_i32 = arith.constant 5 : i32
    %0 = arith.muli %arg0, %c5_i32 : i32
    %1 = arith.addi %0, %0 : i32
    return %1 : i32
  }
}

