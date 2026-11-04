module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c1 = arith.constant 1 : index
    %c4096_i32 = arith.constant 4096 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4096], strides: [%c1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [4096], strides: [%c1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4096], strides: [%c1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = tensor.empty() : tensor<4096xf32>
    %2 = tensor.empty() : tensor<4096xf32>
    %3 = arith.minsi %arg6, %c4096_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %subview = memref.subview %reinterpret_cast[0] [%4] [1] : memref<4096xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<4096xf32>
    %subview_2 = memref.subview %alloc[0] [%4] [1] : memref<4096xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_2 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    %5 = bufferization.to_tensor %subview_2 restrict : memref<?xf32, strided<[1]>>
    %inserted_slice = tensor.insert_slice %5 into %0[0] [%4] [1] : tensor<?xf32> into tensor<4096xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%4] [1] : memref<4096xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    %alloc_4 = memref.alloc() : memref<4096xf32>
    %subview_5 = memref.subview %alloc_4[0] [%4] [1] : memref<4096xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_5 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    %6 = bufferization.to_tensor %subview_5 restrict : memref<?xf32, strided<[1]>>
    %inserted_slice_6 = tensor.insert_slice %6 into %1[0] [%4] [1] : tensor<?xf32> into tensor<4096xf32>
    %7 = hivm.hir.vadd ins(%inserted_slice, %inserted_slice_6 : tensor<4096xf32>, tensor<4096xf32>) outs(%2 : tensor<4096xf32>) -> tensor<4096xf32>
    %extracted_slice = tensor.extract_slice %7[0] [%4] [1] : tensor<4096xf32> to tensor<?xf32>
    %subview_7 = memref.subview %reinterpret_cast_0[0] [%4] [1] : memref<4096xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xf32>, memref<?xf32, strided<[1]>>) -> ()
    return
  }
}