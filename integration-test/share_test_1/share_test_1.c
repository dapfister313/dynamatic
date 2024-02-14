#include "share_test_1.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int share_test_1(in_int_t a[1000], in_int_t b[1000]) {
  int i;
  int tmp = 0;

For_Loop1:
  for (i = 0; i < 1000; i++) {
    tmp += a[i] * a[i] * b[999 - i]; // * b[999 - i] * 787879999;
  }
/*
int x = 2;
int y = 5;
*/
/*
For_Loop2:
  for (i = 0; i < 1000; i++) {
    //a[i*i] = b[i]*i;
    //a[i*i] = i*b[i];
    //a[0] = i*b[i]*a[i*i];
    tmp += a[999 - i] * b[i];
  }
*/

  return tmp;
}

int main(void) {
  in_int_t a[1000];
  in_int_t b[1000];
  inout_int_t out[1000];

  srand(13);
  for (int j = 0; j < 1000; ++j) {
    a[j] = rand() % 100;
    b[j] = rand() % 100;
  }

  CALL_KERNEL(share_test_1, a, b);
  return 0;
}

// SEPARATOR_FOR_MAIN
