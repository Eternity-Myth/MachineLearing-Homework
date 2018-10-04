clear
clc
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
y=[ 2.94, 4.53, 5.96, 7.88, 9.02, 10.94,  12.14, 13.96, 14.74, 16.68, 17.79, 19.67, 21.20, 22.07, 23.75,  25.22, 27.17,  28.84, 29.84, 31.78];
plot(x,y,'.','MarkerSize',15); %以散点图的形式画出该20组数据点
hold on %不清除图像，使散点图和直线图在同一图中绘制出来
parameter=polyfit(x,y,1); %调用polyfit函数对数据拟合，由于是线性，因此多项式的阶设为1
k=parameter(1); %斜率k
b=parameter(2); %截距b
line=k*x+b;
plot(x,line); %绘制拟合直线
str=strcat('y=',num2str(k),'x+',num2str(b));
text(4,25,str,'interpreter','latex','fontsize',30) %使用text函数在图上标注出直线方程