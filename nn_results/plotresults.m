clc;
clear;
close all;

C =[49.076
36.175
32.995
31.358
30.372
29.544
28.900
28.581
28.125
27.896
27.305
27.468
27.059
26.833
26.934
26.885
26.479
26.288
26.258
26.364
26.022
26.180
25.831
25.842
25.774
25.876
25.735
25.642
25.598
25.646];
E = 1:30;
figure(1)
plot(E,C);
xlabel('Epoch');
ylabel('Classification Error (%)');
title('Training Set (1-layer NN with 100 hidden units)');

C=[39.529
35.850
33.947
33.815
33.972
33.235
32.368
32.743
32.691
31.555
33.155
32.409
32.261
33.140
32.044
32.393
31.135
33.330
32.164
31.287
31.942
32.331
31.205
31.020
32.199
31.637
31.377
32.741
31.017
31.215];
figure(2)
plot(E,C);
xlabel('Epoch');
ylabel('Classification Error (%)');
title('Validation Set (1-layer NN with 100 hidden units)');

C=[39.529
35.850
33.947
33.815
33.972
33.235
32.368
32.743
32.691
31.555
33.155
32.409
32.261];
figure(3)
E=1:size(C,1);
plot(E,C);
xlabel('Epoch');
ylabel('Classification Error (%)');
title('Validation Set with Early Stopping (1-layer NN with 100 hidden units)');

C=[38.889
35.468
33.205
32.199
32.635
32.164
29.080
31.287
31.033
30.258];
figure(4)
E=1:size(C,1);
plot(E,C);
xlabel('Epoch');
ylabel('Classification Error (%)');
title('Validation Set with Early Stopping (1-layer NN with 300 hidden units)');

C=[41.397
35.540
33.318
31.254
31.971
32.955
28.400
29.219
29.694
29.244];
figure(5)
E=1:size(C,1);
plot(E,C);
xlabel('Epoch');
ylabel('Classification Error (%)');
title('Validation Set with Early Stopping (1-layer NN with 500 hidden units)');

C=[49.616
40.130
34.984
32.750
32.091
29.447
28.900
27.419
29.150
28.110
27.102
26.795
26.494
26.675
26.857
26.834];
figure(6)
E=1:size(C,1);
plot(E,C);
xlabel('Epoch');
ylabel('Classification Error (%)');
title('Validation Set with Early Stopping (2-layer NN with 300 hidden units)');
