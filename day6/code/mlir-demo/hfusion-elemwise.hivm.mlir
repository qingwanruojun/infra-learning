module {
  func.func @test_elemwise_unary_ops(%arg0: memref<6x6xf32>, %arg1: memref<6x6xf32>) {
    hivm.hir.vexp ins(%arg0 : memref<6x6xf32>) outs(%arg1 : memref<6x6xf32>)
    hivm.hir.vabs ins(%arg0 : memref<6x6xf32>) outs(%arg1 : memref<6x6xf32>)
    hivm.hir.vln ins(%arg0 : memref<6x6xf32>) outs(%arg1 : memref<6x6xf32>)
    return
  }
}

