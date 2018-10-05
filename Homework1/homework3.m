clear all
clc
dataset=[1	0.697 	0.460 	1;
    2 	0.774 	0.376 	1;
    3 	0.634 	0.264 	1;
    4 	0.608 	0.318 	1;
    5 	0.556 	0.215 	1;
    6 	0.403 	0.237 	1;
    7 	0.481 	0.149 	1;
    8 	0.437 	0.211 	1;
    9 	0.666 	0.091 	0;
    10 	0.243 	0.267 	0;
    11 	0.245 	0.057 	0;
    12 	0.343 	0.099 	0;
    13 	0.639 	0.161 	0;
    14 	0.657 	0.198 	0;
    15 	0.360 	0.370 	0;
    16 	0.593 	0.042 	0;
    17 	0.719 	0.103 	0];
X=dataset(:,(2:3));%X取密度与含糖率两个属性
Y=dataset(:,4);%1表示是好瓜，0表示不是好瓜

E=0;
for times=1:10
    posrand=randperm(8);
    negrand=randperm(8,8)+8;
    %去掉最后一个样本采用4折交叉验证，一共需要划分4个子集
    %去掉最后一个样本后，正负例均为8个，因此每个子集应包含正负例各2个
    subset1=[dataset(posrand(1,1:2),2:4);dataset(negrand(1,1:2),2:4)];%子集1
    subset2=[dataset(posrand(1,3:4),2:4);dataset(negrand(1,3:4),2:4)];%子集2
    subset3=[dataset(posrand(1,5:6),2:4);dataset(negrand(1,5:6),2:4)];%子集3
    subset4=[dataset(posrand(1,7:8),2:4);dataset(negrand(1,7:8),2:4)];%子集4

    %绘制正负例分布图
    figure;
    hold on;
    pos=find(Y==1);%正例
    neg=find(Y==0);%负例
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%画正例点，用“+”表示
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%画负例点，用“。”表示
    xlabel('密度')
    ylabel('含糖率')

    training1=[subset1;subset2;subset3];%第一次，取前3个子集作训练集，第4个子集作测试集
    X1=training1(:,1:2);%X取密度与含糖率两个属性
    Y1=training1(:,3);%1表示是好瓜，0表示不是好瓜
    [m,n]=size(X1);
    X1=[X1,ones(m,1)];%在最右端添加1列1以拟合常数项b
    initial_theta=zeros(n+1,1);%初始化系数theta
    options=optimset('GradObj','on','MaxIter',400);
    %调用fminunc函数求解逻辑回归的最佳参数，也就是使costFunction达到最小值的对应参数theta
    [theta1,cost1]=fminunc(@(t)(costFunction(t, X1, Y1)),initial_theta,options);
    %第4个子集用于测试
    X4=subset4(:,1:2);
    X4=[X4,ones(4,1)];
    %计算正确率
    count=0;
    for i=1:4
        if((X4(i,:)*theta1>0&&subset4(i,3)==1)||(X4(i,:)*theta1<0&&subset4(i,3)==0))
            count=count+1;
        end
    end
    e1=count/4;

    %绘制判定边界直线
    x=0:0.1:0.8;
    line=(-theta1(3,1)-theta1(1,1)*x)/theta1(2,1);%判定边界直线方程为：theta1*x1+theta2*x2+theta3=0，可以反解出theta2的斜截式方程
    plot(x,line);
    str=strcat('y=',num2str(-theta1(1,1)/theta1(2,1)),'x+',num2str(-theta1(3,1)/theta1(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %使用text函数在图上标注出直线方程


    %绘制正负例分布图
    figure;
    hold on;
    pos=find(Y==1);%正例
    neg=find(Y==0);%负例
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%画正例点，用“+”表示
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%画负例点，用“。”表示
    xlabel('密度')
    ylabel('含糖率')

    training2=[subset1;subset2;subset4];%第二次，取1 2 4子集作训练集，第3个子集作测试集
    X2=training2(:,1:2);%X取密度与含糖率两个属性
    Y2=training2(:,3);%1表示是好瓜，0表示不是好瓜
    [m,n]=size(X2);
    X2=[X2,ones(m,1)];%在最右端添加1列1以拟合常数项b
    initial_theta=zeros(n+1,1);%初始化系数theta
    options=optimset('GradObj','on','MaxIter',400);
    %调用fminunc函数求解逻辑回归的最佳参数，也就是使costFunction达到最小值的对应参数theta
    [theta2,cost2]=fminunc(@(t)(costFunction(t, X2, Y2)),initial_theta,options);
    %第3个子集用于测试
    X3=subset3(:,1:2);
    X3=[X3,ones(4,1)];
    %计算正确率
    count=0;
    for i=1:4
        if((X3(i,:)*theta2>0&&subset3(i,3)==1)||(X3(i,:)*theta2<0&&subset3(i,3)==0))
            count=count+1;
        end
    end
    e2=count/4;

    %绘制判定边界直线
    x=0:0.1:0.8;
    line=(-theta2(3,1)-theta2(1,1)*x)/theta2(2,1);%判定边界直线方程为：theta1*x1+theta2*x2+theta3=0，可以反解出theta2的斜截式方程
    plot(x,line);
    str=strcat('y=',num2str(-theta2(1,1)/theta2(2,1)),'x+',num2str(-theta2(3,1)/theta2(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %使用text函数在图上标注出直线方程

    %绘制正负例分布图
    figure;
    hold on;
    pos=find(Y==1);%正例
    neg=find(Y==0);%负例
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%画正例点，用“+”表示
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%画负例点，用“。”表示
    xlabel('密度')
    ylabel('含糖率')

    training3=[subset1;subset3;subset4];%第三次，取1 3 4作训练集，第2个子集作测试集
    X3=training3(:,1:2);%X取密度与含糖率两个属性
    Y3=training3(:,3);%1表示是好瓜，0表示不是好瓜
    [m,n]=size(X3);
    X3=[X3,ones(m,1)];%在最右端添加1列1以拟合常数项b
    initial_theta=zeros(n+1,1);%初始化系数theta
    options=optimset('GradObj','on','MaxIter',400);
    %调用fminunc函数求解逻辑回归的最佳参数，也就是使costFunction达到最小值的对应参数theta
    [theta3,cost3]=fminunc(@(t)(costFunction(t, X3, Y3)),initial_theta,options);
    %第2个子集用于测试
    X2=subset2(:,1:2);
    X2=[X2,ones(4,1)];
    %计算正确率
    count=0;
    for i=1:4
        if((X2(i,:)*theta3>0&&subset2(i,3)==1)||(X2(i,:)*theta3<0&&subset2(i,3)==0))
            count=count+1;
        end
    end
    e3=count/4;

    %绘制判定边界直线
    x=0:0.1:0.8;
    line=(-theta3(3,1)-theta3(1,1)*x)/theta3(2,1);%判定边界直线方程为：theta1*x1+theta2*x2+theta3=0，可以反解出theta2的斜截式方程
    plot(x,line);
    str=strcat('y=',num2str(-theta3(1,1)/theta3(2,1)),'x+',num2str(-theta3(3,1)/theta3(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %使用text函数在图上标注出直线方程

    %绘制正负例分布图
    figure;
    hold on;
    pos=find(Y==1);%正例
    neg=find(Y==0);%负例
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%画正例点，用“+”表示
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%画负例点，用“。”表示
    xlabel('密度')
    ylabel('含糖率')

    training4=[subset2;subset3;subset4];%第四次，取2 3 4作训练集，第1个子集作测试集
    X4=training4(:,1:2);%X取密度与含糖率两个属性
    Y4=training4(:,3);%1表示是好瓜，0表示不是好瓜
    [m,n]=size(X4);
    X4=[X4,ones(m,1)];%在最右端添加1列1以拟合常数项b
    initial_theta=zeros(n+1,1);%初始化系数theta
    options=optimset('GradObj','on','MaxIter',400);
    %调用fminunc函数求解逻辑回归的最佳参数，也就是使costFunction达到最小值的对应参数theta
    [theta4,cost4]=fminunc(@(t)(costFunction(t, X4, Y4)),initial_theta,options);
    %第1个子集用于测试
    X1=subset1(:,1:2);
    X1=[X1,ones(4,1)];
    %计算正确率
    count=0;
    for i=1:4
        if((X1(i,:)*theta4>0&&subset1(i,3)==1)||(X1(i,:)*theta4<0&&subset1(i,3)==0))
            count=count+1;
        end
    end
    e4=count/4;

    %绘制判定边界直线
    x=0:0.1:0.8;
    line=(-theta4(3,1)-theta4(1,1)*x)/theta4(2,1);%判定边界直线方程为：theta1*x1+theta2*x2+theta3=0，可以反解出theta2的斜截式方程
    plot(x,line);
    str=strcat('y=',num2str(-theta4(1,1)/theta4(2,1)),'x+',num2str(-theta4(3,1)/theta4(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %使用text函数在图上标注出直线方程

    averageE=(e1+e2+e3+e4)/4;
    E=E+averageE;
end
E/10

function [J, grad] = costFunction(theta, X, Y)
m = length(Y); %训练集数据数
grad = zeros(size(theta));%初始化梯度为0
h=1.0./(1.0+exp(-1*X*theta));%对数几率函数，即Sigmoid函数
m=size(Y,1);
J=((-1*Y)'*log(h)-(1-Y)'*log(1-h))/m;%简化后统一形式的costFunction J
for i=1:size(theta,1),
    grad(i)=((h-Y)'*X(:,i))/m;%对costFunction J求导可得梯度的表达式
end
end