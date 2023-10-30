// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-infer-basic-blocks --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @backtrackToArgument(
// CHECK-SAME:                                        %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]] = fork [1] %[[VAL_0]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_2:.*]] = fork [1] %[[VAL_1]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_3:.*]] = fork [1] %[[VAL_2]] {bb = 0 : ui32} : none
// CHECK:           end
// CHECK:         }
handshake.func @backtrackToArgument(%start: none) {
  %0 = fork [1] %start : none
  %1 = fork [1] %0 : none
  %2 = fork [1] %1 : none
  end
}

// -----

// CHECK-LABEL:   handshake.func @backtrackToKnownBB(
// CHECK-SAME:                                       %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_1]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_2]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_4:.*]] = fork [1] %[[VAL_3]]#0 {bb = 1 : ui32} : none
// CHECK:           %[[VAL_5:.*]] = fork [1] %[[VAL_3]]#1 {bb = 1 : ui32} : none
// CHECK:           end
// CHECK:         }
handshake.func @backtrackToKnownBB(%start: none) {
  %0 = br %start {bb = 0 : ui32} : none
  %1 = merge %0 {bb = 1 : ui32} : none
  %2:2 = fork [2] %1 : none
  %3 = fork [1] %2#0 : none
  %4 = fork [1] %2#1 : none
  end
}

// -----

// CHECK-LABEL:   handshake.func @backtrackConflict(
// CHECK-SAME:                                      %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_2:.*]] = br %[[VAL_0]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_1]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_2]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_3]], %[[VAL_4]] : none
// CHECK:           end
// CHECK:         }
handshake.func @backtrackConflict(%start: none) {
  %0 = br %start {bb = 0 : ui32} : none
  %1 = br %start {bb = 0 : ui32} : none
  %2 = merge %0 {bb = 1 : ui32} : none
  %3 = merge %1 {bb = 2 : ui32} : none
  %4 = merge %2, %3 : none
  end
}

// -----

// CHECK-LABEL:   handshake.func @forwardToKnownBB(
// CHECK-SAME:                                     %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]]:2 = fork [2] %[[VAL_0]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_1]]#0 {bb = 1 : ui32} : none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_1]]#1 {bb = 1 : ui32} : none
// CHECK:           end
// CHECK:         }
handshake.func @forwardToKnownBB(%start: none) {
  %1:2 = fork [2] %start : none
  %2 = merge %1#0 {bb = 1 : ui32} : none
  %3 = merge %1#1 {bb = 1 : ui32} : none
  end
}

// -----

// CHECK-LABEL:   handshake.func @forwardConflict(
// CHECK-SAME:                                    %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]] = merge %[[VAL_0]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_1]], %[[VAL_2]] : none
// CHECK:           %[[VAL_4:.*]]:2 = fork [2] %[[VAL_3]] : none
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_4]]#0 {bb = 1 : ui32} : none
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_4]]#1 {bb = 2 : ui32} : none
// CHECK:           end
// CHECK:         }
handshake.func @forwardConflict(%start: none) {
  %0 = merge %start {bb = 1 : ui32} : none
  %1 = merge %start {bb = 2 : ui32} : none
  %2 = merge %0, %1 : none
  %3:2 = fork [2] %2 : none
  %4 = merge %3#0 {bb = 1 : ui32} : none
  %5 = merge %3#1 {bb = 2 : ui32} : none
  end
}

// -----

// CHECK-LABEL:   handshake.func @forwardOverBackward(
// CHECK-SAME:                                        %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_1]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_2]]#0 {bb = 2 : ui32} : none
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_2]]#1 {bb = 2 : ui32} : none
// CHECK:           end
// CHECK:         }
handshake.func @forwardOverBackward(%start: none) {
  %0 = br %start {bb = 1 : ui32} : none
  %1:2 = fork [2] %0 : none
  %2 = merge %1#0 {bb = 2 : ui32} : none
  %3 = merge %1#1 {bb = 2 : ui32} : none
  end
}

// -----

// CHECK-LABEL:   handshake.func @backwardOverForward(
// CHECK-SAME:                                        %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_1]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_2]]#0 {bb = 2 : ui32} : none
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_2]]#1 {bb = 3 : ui32} : none
// CHECK:           end
// CHECK:         }
handshake.func @backwardOverForward(%start: none) {
  %0 = br %start {bb = 1 : ui32} : none
  %1:2 = fork [2] %0 : none
  %2 = merge %1#0 {bb = 2 : ui32} : none
  %3 = merge %1#1 {bb = 3 : ui32} : none
  end
}
