#include<stdio.h>
#include<math.h>
int main(){
    double s;
    int n;
    printf("Nhap n:");
    scanf("%d ",&n);
    s = n;
    for(int i = n;i >= 1;i--){
        s = sqrt(i + sqrt(s));
    }
    printf("\ndap an: %.5lf",s);
    return 0;
}