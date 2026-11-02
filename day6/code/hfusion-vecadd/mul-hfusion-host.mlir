// Day6 negative control: same HFusion/Linalg-level HOST function and same
// fusing/scheduling/outlining path, but computes a*b instead of a+b.
// The device ELF it produces must be rejected by the fixed host validator.
//
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @add(%arg0: tensor<16xi16>, %arg1: tensor<16xi16>) -> tensor<16xi16>
      attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %empty = tensor.empty() : tensor<16xi16>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
      ins(%arg0, %arg1 : tensor<16xi16>, tensor<16xi16>)
      outs(%empty : tensor<16xi16>) -> tensor<16xi16>
    return %sum : tensor<16xi16>
  }
}
