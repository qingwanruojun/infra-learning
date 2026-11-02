// Day6 full-chain experiment: a simple HFusion/Linalg-level HOST function.
// The HFusion pipeline fuses/schedules/outlines it into a device kernel, then
// ConvertToHIVM + HIVM optimization + hivmc produce the device ELF.
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @add(%arg0: tensor<16xi16>, %arg1: tensor<16xi16>) -> tensor<16xi16>
      attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %empty = tensor.empty() : tensor<16xi16>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%arg0, %arg1 : tensor<16xi16>, tensor<16xi16>)
      outs(%empty : tensor<16xi16>) -> tensor<16xi16>
    return %sum : tensor<16xi16>
  }
}
