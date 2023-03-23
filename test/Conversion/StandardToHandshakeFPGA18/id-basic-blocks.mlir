// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --lower-std-to-handshake-fpga18="id-basic-blocks" %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @memInterfaceNoID(
// CHECK-SAME:                                     %[[VAL_0:.*]]: memref<4xi32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: index,
// CHECK-SAME:                                     %[[VAL_2:.*]]: none, ...) -> i32 attributes {argNames = ["in0", "in1", "in2"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = mem_controller{{\[}}%[[VAL_0]] : memref<4xi32>] (%[[VAL_5:.*]]) {accesses = {{\[\[}}#handshake<AccessType Load>]], id = 0 : i32} : (index) -> (i32, none)
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_1]] {bb = 0 : ui32} : index
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_2]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_5]], %[[VAL_8:.*]] = d_load{{\[}}%[[VAL_6]]] %[[VAL_3]] {bb = 0 : ui32} : index, i32
// CHECK:           %[[VAL_9:.*]] = d_return {bb = 0 : ui32} %[[VAL_8]] : i32
// CHECK:           end {bb = 0 : ui32} %[[VAL_9]], %[[VAL_4]] : i32, none
// CHECK:         }
func.func @memInterfaceNoID(%arg0: memref<4xi32>, %arg1: index) -> i32 {
  %0 = memref.load %arg0[%arg1] : memref<4xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   handshake.func @ifThenElse(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32,
// CHECK-SAME:                               %[[VAL_1:.*]]: i1,
// CHECK-SAME:                               %[[VAL_2:.*]]: none, ...) -> i32 attributes {argNames = ["in0", "in1", "in2"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_1]] {bb = 0 : ui32} : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_2]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_4]], %[[VAL_3]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_4]], %[[VAL_5]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_6]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = control_merge %[[VAL_8]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_11]] {bb = 1 : ui32, value = 1 : i32} : i32
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_11]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_15]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_18:.*]] = merge %[[VAL_7]] {bb = 2 : ui32} : i32
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = control_merge %[[VAL_9]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_22:.*]] = constant %[[VAL_19]] {bb = 2 : ui32, value = 2 : i32} : i32
// CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_18]], %[[VAL_22]] {bb = 2 : ui32} : i32
// CHECK:           %[[VAL_24:.*]] = br %[[VAL_19]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_25:.*]] = br %[[VAL_23]] {bb = 2 : ui32} : i32
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_27:.*]] {{\[}}%[[VAL_25]], %[[VAL_17]]] {bb = 3 : ui32} : index, i32
// CHECK:           %[[VAL_28:.*]], %[[VAL_27]] = control_merge %[[VAL_24]], %[[VAL_16]] {bb = 3 : ui32} : none
// CHECK:           %[[VAL_29:.*]] = d_return {bb = 3 : ui32} %[[VAL_26]] : i32
// CHECK:           end {bb = 3 : ui32} %[[VAL_29]] : i32
// CHECK:         }
func.func @ifThenElse(%arg0: i32, %arg1: i1) -> i32 {
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %res1 = arith.addi %arg0, %c1 : i32
  cf.br ^bb3(%res1 : i32)
^bb2:
  %c2 = arith.constant 2 : i32
  %res2 = arith.addi %arg0, %c2 : i32
  cf.br ^bb3(%res2 : i32)
^bb3(%res : i32):
  return %res : i32
}

// -----

// CHECK-LABEL:   handshake.func @multipleReturns(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i1,
// CHECK-SAME:                                    %[[VAL_2:.*]]: none, ...) -> i32 attributes {argNames = ["in0", "in1", "in2"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_1]] {bb = 0 : ui32} : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_2]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_4]], %[[VAL_3]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_4]], %[[VAL_5]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_6]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = control_merge %[[VAL_8]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_11]] {bb = 1 : ui32, value = 1 : i32} : i32
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_16:.*]] = d_return {bb = 1 : ui32} %[[VAL_15]] : i32
// CHECK:           %[[VAL_17:.*]] = merge %[[VAL_7]] {bb = 2 : ui32} : i32
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = control_merge %[[VAL_9]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_21:.*]] = constant %[[VAL_18]] {bb = 2 : ui32, value = 2 : i32} : i32
// CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_17]], %[[VAL_21]] {bb = 2 : ui32} : i32
// CHECK:           %[[VAL_23:.*]] = d_return {bb = 2 : ui32} %[[VAL_22]] : i32
// CHECK:           %[[VAL_24:.*]] = merge %[[VAL_16]], %[[VAL_23]] {bb = 3 : ui32} : i32
// CHECK:           end {bb = 3 : ui32} %[[VAL_24]] : i32
// CHECK:         }
func.func @multipleReturns(%arg0: i32, %arg1: i1) -> i32 {
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %res1 = arith.addi %arg0, %c1 : i32
  return %res1 : i32
^bb2:
  %c2 = arith.constant 2 : i32
  %res2 = arith.addi %arg0, %c2 : i32
  return %res2 : i32
}

// -----

// CHECK-LABEL:   handshake.func @simpleLoop(
// CHECK-SAME:                               %[[VAL_0:.*]]: index,
// CHECK-SAME:                               %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] {bb = 0 : ui32} : index
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_1]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_3]] {bb = 0 : ui32, value = 0 : index} : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_3]] {bb = 0 : ui32, value = 1 : index} : index
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_2]] {bb = 0 : ui32} : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_3]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_5]] {bb = 0 : ui32} : index
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_7]] {bb = 0 : ui32} : index
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_13:.*]] {{\[}}%[[VAL_14:.*]], %[[VAL_10]]] {bb = 1 : ui32} : index, index
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_13]] {{\[}}%[[VAL_16:.*]], %[[VAL_8]]] {bb = 1 : ui32} : index, index
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_13]] {{\[}}%[[VAL_18:.*]], %[[VAL_11]]] {bb = 1 : ui32} : index, index
// CHECK:           %[[VAL_19:.*]], %[[VAL_13]] = control_merge %[[VAL_20:.*]], %[[VAL_9]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_15]] {bb = 1 : ui32} : index
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = cond_br %[[VAL_21]], %[[VAL_12]] {bb = 1 : ui32} : index
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = cond_br %[[VAL_21]], %[[VAL_15]] {bb = 1 : ui32} : index
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = cond_br %[[VAL_21]], %[[VAL_17]] {bb = 1 : ui32} : index
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_21]], %[[VAL_19]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_30:.*]] = merge %[[VAL_24]] {bb = 2 : ui32} : index
// CHECK:           %[[VAL_31:.*]] = merge %[[VAL_26]] {bb = 2 : ui32} : index
// CHECK:           %[[VAL_32:.*]] = merge %[[VAL_22]] {bb = 2 : ui32} : index
// CHECK:           %[[VAL_33:.*]], %[[VAL_34:.*]] = control_merge %[[VAL_28]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_35:.*]] = arith.addi %[[VAL_32]], %[[VAL_31]] {bb = 2 : ui32} : index
// CHECK:           %[[VAL_16]] = br %[[VAL_30]] {bb = 2 : ui32} : index
// CHECK:           %[[VAL_18]] = br %[[VAL_31]] {bb = 2 : ui32} : index
// CHECK:           %[[VAL_20]] = br %[[VAL_33]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_14]] = br %[[VAL_35]] {bb = 2 : ui32} : index
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = control_merge %[[VAL_29]] {bb = 3 : ui32} : none
// CHECK:           %[[VAL_38:.*]] = d_return {bb = 3 : ui32} %[[VAL_36]] : none
// CHECK:           end {bb = 3 : ui32} %[[VAL_38]] : none
// CHECK:         }
func.func @simpleLoop(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0 : index)
^bb1(%1: index):
  %cmp = arith.cmpi slt, %1, %arg0 : index
  cf.cond_br %cmp, ^bb2, ^bb3
^bb2:
  %2 = arith.addi %1, %c1 : index
  cf.br ^bb1(%2 : index)
^bb3:
  return
}
