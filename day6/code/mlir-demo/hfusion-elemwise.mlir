// Day6 step5: HFusion-level input (linalg named ops extended by HFusion).
// Copied from bishengir/test/Conversion/HFusionToHIVM/hfusion-to-hivm.mlir.
func.func @test_elemwise_unary_ops(
  %src : memref<6x6xf32>, %dst : memref<6x6xf32>) {
  linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)
  return
}
