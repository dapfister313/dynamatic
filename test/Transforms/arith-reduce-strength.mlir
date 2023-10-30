// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --arith-reduce-strength="max-adder-depth-mul=3" --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @replaceMulAddWithSub(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : i32
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }
func.func @replaceMulAddWithSub(%arg0: i32, %arg1: i32) -> i32 {
  %negOne = arith.constant -1 : i32
  %mul = arith.muli %negOne, %arg0 : i32
  %add = arith.addi %mul, %arg1 : i32
  return %add : i32
}

// -----

// CHECK-LABEL:   func.func @replaceMulAddWithSubWrongCst(
// CHECK-SAME:                                            %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                            %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant -2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @replaceMulAddWithSubWrongCst(%arg0: i32, %arg1: i32) -> i32 {
  %negTwo = arith.constant -2 : i32
  %mul = arith.muli %negTwo, %arg0 : i32
  %add = arith.addi %mul, %arg1 : i32
  return %add : i32
}

// -----

// CHECK-LABEL:   func.func @replaceMulAddWithSubNoCst(
// CHECK-SAME:                                         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                         %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_2]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @replaceMulAddWithSubNoCst(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %mul = arith.muli %arg2, %arg0 : i32
  %add = arith.addi %mul, %arg1 : i32
  return %add : i32
}

// -----

// CHECK-LABEL:   func.func @reduceStrengthMul3(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.shli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_0]], %[[VAL_2]] : i32
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
func.func @reduceStrengthMul3(%arg0: i32) -> i32 {
  %cst = arith.constant 3 : i32
  %mul = arith.muli %arg0, %cst : i32
  return %mul : i32
}

// -----

// CHECK-LABEL:   func.func @reduceStrengthMul30(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_5:.*]] = arith.shli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = arith.shli %[[VAL_0]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = arith.shli %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.shli %[[VAL_0]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_7]], %[[VAL_10]] : i32
// CHECK:           return %[[VAL_11]] : i32
// CHECK:         }
func.func @reduceStrengthMul30(%arg0: i32) -> i32 {
  %cst = arith.constant 30 : i32
  %mul = arith.muli %arg0, %cst : i32
  return %mul : i32
}

// -----

// CHECK-LABEL:   func.func @reduceStrengthMulTooDeep(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 511 : i32
// CHECK:           %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }
func.func @reduceStrengthMulTooDeep(%arg0: i32) -> i32 {
  %cst = arith.constant 511 : i32
  %mul = arith.muli %arg0, %cst : i32
  return %mul : i32
}

// -----

// CHECK-LABEL:   func.func @reduceStrengthMulPowTwo(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 9 : i32
// CHECK:           %[[VAL_2:.*]] = arith.shli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }
func.func @reduceStrengthMulPowTwo(%arg0: i32) -> i32 {
  %cst = arith.constant 512 : i32
  %mul = arith.muli %arg0, %cst : i32
  return %mul : i32
}

// -----

// CHECK-LABEL:   func.func @promoteCmpWrongType(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                   %[[VAL_1:.*]]: i32) -> i1 {
// CHECK:           %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }
func.func @promoteCmpWrongType(%arg0: i32, %arg1: i32) -> i1 {
  %cmp = arith.cmpi slt, %arg0, %arg1 : i32
  return %cmp : i1
}

// -----

// CHECK-LABEL:   func.func @promoteCmpWrongPred(
// CHECK-SAME:                                   %[[VAL_0:.*]]: index,
// CHECK-SAME:                                   %[[VAL_1:.*]]: index) -> i1 {
// CHECK:           %[[VAL_2:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : index
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }
func.func @promoteCmpWrongPred(%arg0: index, %arg1: index) -> i1 {
  %cmp = arith.cmpi eq, %arg0, %arg1 : index
  return %cmp : i1
}
